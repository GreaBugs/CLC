from copy import deepcopy
import math

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from detectron2.config import configurable
from torch import nn
from torch.nn import functional as F

from fastinst.utils.misc import nested_tensor_from_tensor_list
from .utils import TRANSFORMER_DECODER_REGISTRY, QueryProposal, \
    CrossAttentionLayer, SelfAttentionLayer, FFNLayer, MLP


@TRANSFORMER_DECODER_REGISTRY.register()
class FastInstDecoder(nn.Module):

    @configurable
    def __init__(
            self,
            in_channels,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            num_aux_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int
    ):
        """
        Args:
            in_channels: channels of the input features
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            num_aux_queries: number of auxiliary queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
        """
        super().__init__()
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.num_queries = num_queries
        self.num_aux_queries = num_aux_queries
        self.criterion = None

        meta_pos_size = int(round(math.sqrt(self.num_queries)))
        self.meta_pos_embed = nn.Parameter(torch.empty(1, hidden_dim, meta_pos_size, meta_pos_size))
        if num_aux_queries > 0:
            self.empty_query_features = nn.Embedding(num_aux_queries, hidden_dim)
            self.empty_query_pos_embed = nn.Embedding(num_aux_queries, hidden_dim)

        self.query_proposal = QueryProposal(hidden_dim, num_queries, num_classes)

        self.transformer_query_cross_attention_layers = nn.ModuleList()
        self.transformer_query_self_attention_layers = nn.ModuleList()
        self.transformer_query_ffn_layers = nn.ModuleList()
        self.transformer_mask_cross_attention_layers = nn.ModuleList()
        self.transformer_mask_ffn_layers = nn.ModuleList()
        for idx in range(self.num_layers):
            self.transformer_query_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_query_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_query_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_mask_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_mask_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm
                )
            )

        self.decoder_query_norm_layers = nn.ModuleList()
        self.class_embed_layers = nn.ModuleList()
        self.mask_embed_layers = nn.ModuleList()
        self.mask_features_layers = nn.ModuleList()
        for idx in range(self.num_layers + 1):
            self.decoder_query_norm_layers.append(nn.LayerNorm(hidden_dim))
            self.class_embed_layers.append(MLP(hidden_dim, hidden_dim, num_classes + 1, 3))
            self.mask_embed_layers.append(MLP(hidden_dim, hidden_dim, mask_dim, 3))
            self.mask_features_layers.append(nn.Linear(hidden_dim, mask_dim))

    @classmethod
    def from_config(cls, cfg, in_channels, input_shape):
        ret = {}
        ret["in_channels"] = in_channels

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.FASTINST.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.FASTINST.NUM_OBJECT_QUERIES
        ret["num_aux_queries"] = cfg.MODEL.FASTINST.NUM_AUX_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.FASTINST.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.FASTINST.DIM_FEEDFORWARD

        ret["dec_layers"] = cfg.MODEL.FASTINST.DEC_LAYERS
        ret["pre_norm"] = cfg.MODEL.FASTINST.PRE_NORM

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret

    def forward(self, x, mask_features, targets=None):
        bs = x[0].shape[0]
        proposal_size = x[1].shape[-2:]
        pixel_feature_size = x[2].shape[-2:]

        pixel_pos_embeds = F.interpolate(self.meta_pos_embed, size=pixel_feature_size,
                                         mode="bilinear", align_corners=False)
        proposal_pos_embeds = F.interpolate(self.meta_pos_embed, size=proposal_size,
                                            mode="bilinear", align_corners=False)

        pixel_features = x[2].flatten(2).permute(2, 0, 1)
        pixel_pos_embeds = pixel_pos_embeds.flatten(2).permute(2, 0, 1)

        query_features, query_pos_embeds, query_locations, proposal_cls_logits = self.query_proposal(
            x[1], proposal_pos_embeds
        )
        query_features = query_features.permute(2, 0, 1)
        query_pos_embeds = query_pos_embeds.permute(2, 0, 1)
        if self.num_aux_queries > 0:
            aux_query_features = self.empty_query_features.weight.unsqueeze(1).repeat(1, bs, 1)
            aux_query_pos_embed = self.empty_query_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            query_features = torch.cat([query_features, aux_query_features], dim=0)
            query_pos_embeds = torch.cat([query_pos_embeds, aux_query_pos_embed], dim=0)

        outputs_class, outputs_mask, attn_mask, _, _ = self.forward_prediction_heads(
            query_features, pixel_features, pixel_feature_size, -1, return_attn_mask=True
        )
        predictions_class = [outputs_class]
        predictions_mask = [outputs_mask]
        predictions_matching_index = [None]
        query_feature_memory = [query_features]
        pixel_feature_memory = [pixel_features]

        for i in range(self.num_layers):
            query_features, pixel_features = self.forward_one_layer(
                query_features, pixel_features, query_pos_embeds, pixel_pos_embeds, attn_mask, i
            )
            if i < self.num_layers - 1:
                outputs_class, outputs_mask, attn_mask, _, _ = self.forward_prediction_heads(
                    query_features, pixel_features, pixel_feature_size, i, return_attn_mask=True,
                )
            else:
                outputs_class, outputs_mask, _, matching_indices, gt_attn_mask = self.forward_prediction_heads(
                    query_features, pixel_features, pixel_feature_size, i,
                    return_gt_attn_mask=self.training, targets=targets, query_locations=query_locations
                )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_matching_index.append(None)
            query_feature_memory.append(query_features)
            pixel_feature_memory.append(pixel_features)

        guided_predictions_class = []
        guided_predictions_mask = []
        guided_predictions_matching_index = []
        if self.training:
            for i in range(self.num_layers):
                query_features, pixel_features = self.forward_one_layer(
                    query_feature_memory[i + 1], pixel_feature_memory[i + 1], query_pos_embeds,
                    pixel_pos_embeds, gt_attn_mask, i
                )

                outputs_class, outputs_mask, _, _, _ = self.forward_prediction_heads(
                    query_features, pixel_features, pixel_feature_size, idx_layer=i
                )

                guided_predictions_class.append(outputs_class)
                guided_predictions_mask.append(outputs_mask)
                guided_predictions_matching_index.append(matching_indices)

        predictions_class = guided_predictions_class + predictions_class
        predictions_mask = guided_predictions_mask + predictions_mask
        predictions_matching_index = guided_predictions_matching_index + predictions_matching_index

        out = {
            'proposal_cls_logits': proposal_cls_logits,
            'query_locations': query_locations,
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_matching_indices': predictions_matching_index[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class, predictions_mask, predictions_matching_index, query_locations
            )
        }
        return out

    def forward_one_layer(self, query_features, pixel_features, query_pos_embeds, pixel_pos_embeds, attn_mask, i):
        pixel_features = self.transformer_mask_cross_attention_layers[i](
            pixel_features, query_features, query_pos=pixel_pos_embeds, pos=query_pos_embeds
        )
        pixel_features = self.transformer_mask_ffn_layers[i](pixel_features)

        query_features = self.transformer_query_cross_attention_layers[i](
            query_features, pixel_features, memory_mask=attn_mask, query_pos=query_pos_embeds, pos=pixel_pos_embeds
        )
        query_features = self.transformer_query_self_attention_layers[i](
            query_features, query_pos=query_pos_embeds
        )
        query_features = self.transformer_query_ffn_layers[i](query_features)
        return query_features, pixel_features

    def forward_prediction_heads(self, query_features, pixel_features, pixel_feature_size, idx_layer,
                                 return_attn_mask=False, return_gt_attn_mask=False,
                                 targets=None, query_locations=None):
        decoder_query_features = self.decoder_query_norm_layers[idx_layer + 1](query_features[:self.num_queries])  # query_features(108, bs, 256)
        decoder_query_features = decoder_query_features.transpose(0, 1)  # (1, 100, 256)
        if self.training or idx_layer + 1 == self.num_layers:
            outputs_class = self.class_embed_layers[idx_layer + 1](decoder_query_features)   # (bs, 100, 81)
        else:
            outputs_class = None
        outputs_mask_embed = self.mask_embed_layers[idx_layer + 1](decoder_query_features)  # (1, 100, 256)
        outputs_mask_features = self.mask_features_layers[idx_layer + 1](pixel_features.transpose(0, 1)) # (1, 3744, 256)

        outputs_mask = torch.einsum("bqc,blc->bql", outputs_mask_embed, outputs_mask_features)  # (1, 100, 3744)  这个操作通常用于计算查询与特征之间的相似度、匹配度或注意力权重。通过执行张量乘法，可以将查询与特征进行对应位置的相关性计算，并得到输出张量作为后续任务的结果或输入。
        outputs_mask = outputs_mask.reshape(-1, self.num_queries, *pixel_feature_size)  #  (bs, 100, 52, 72)

        if return_attn_mask:  
            # outputs_mask.shape: b, q, h, w
            attn_mask = F.pad(outputs_mask, (0, 0, 0, 0, 0, self.num_aux_queries), "constant", 1)  # (1, 108, 52, 72)   self.num_aux_queries
            attn_mask = (attn_mask < 0.).flatten(2)  # b, q, hw   (1, 108, 3744)
            invalid_query = attn_mask.all(-1, keepdim=True)  # b, q, 1    (1, 108, 1)
            attn_mask = (~ invalid_query) & attn_mask  # b, q, hw
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)  # (8, 108, 3744)
            attn_mask = attn_mask.detach()
        else:
            attn_mask = None

        if return_gt_attn_mask:
            assert targets is not None and query_locations is not None
            matching_indices = self.criterion.matcher(
                {'pred_logits': outputs_class, 'pred_masks': outputs_mask,
                 'query_locations': query_locations}, targets)   # matching_indice是匹配好的真值索引
            src_idx = self.criterion._get_src_permutation_idx(matching_indices)
            tgt_idx = self.criterion._get_tgt_permutation_idx(matching_indices)

            mask = [t["masks"] for t in targets]  # GT-mask  list[0]:(4, 416, 576)
            #*************************************************
            # for j in range(len(mask)):
            #     for i in range(mask[j].shape[0]):
            #         visual_gt_mask = mask[j][i, :, :]  # [1, 416, 576]
            #         visual_gt_mask = visual_gt_mask.view((mask[j].shape[1], mask[j].shape[2]))  # [416, 576]
            #         visual_gt_mask = visual_gt_mask.cpu()
            #         visual_gt_mask = visual_gt_mask.numpy()
            #         # 将布尔类型的掩码数据转换为整数类型的图像数据
            #         mask_data = np.where(visual_gt_mask, 255, 0).astype(np.uint8)
            #         # 显示图像
            #         plt.imshow(mask_data, cmap='gray')
            #         plt.axis('off')
            #         plt.suptitle('GT_mask')
            #         plt.show()
            #*************************************************
            shen_masks = deepcopy(mask)

            # 边缘提取函数
            def mask_augmentation(shen_masks):
                num_obj, h, w = shen_masks.shape
                shen_masks = shen_masks.cpu()
                result_masks = np.zeros_like(shen_masks, dtype=np.uint8)
                kernel = np.ones((3, 3), dtype=np.uint8)

                # 对每个目标的掩码进行膨胀处理
                for i in range(num_obj):
                    iteration = np.random.randint(1, 2)
                    shen_mask = shen_masks[i]
                    raw_mask = shen_mask.cpu().numpy().astype(np.uint8)
                    p_mask = np.random.rand()
                    if p_mask < 0.7:
                        result_masks[i] = shen_mask
                        # result_masks[i] = cv2.erode(raw_mask, kernel, iterations=iteration1)
                    else:
                        # area = torch.sum(shen_mask)
                        result_masks[i] = cv2.erode(raw_mask, kernel, iterations=iteration)
                return result_masks

            # *********************************************************************
            result_masks_list = []  # erode mask
            for bs_mask in shen_masks:
                bs_augmasks = mask_augmentation(bs_mask)
                bs_augmasks = torch.from_numpy(bs_augmasks)
                bs_augmasks = bs_augmasks.to(torch.bool).to(mask[0].device)
                result_masks_list.append(bs_augmasks)
            # ************************************************
            # 可视化
            # for j in range(len(result_masks_list)):
            #     for i in range(result_masks_list[j].shape[0]):
            #         visual_gt_mask = result_masks_list[j][i, :, :]  # [1, 416, 576]
            #         visual_gt_mask = visual_gt_mask.view((result_masks_list[j].shape[1], result_masks_list[j].shape[2]))  # [416, 576]
            #         visual_gt_mask = visual_gt_mask.cpu()
            #         visual_gt_mask = visual_gt_mask.numpy()
            #         # 将布尔类型的掩码数据转换为整数类型的图像数据
            #         mask_data = np.where(visual_gt_mask, 255, 0).astype(np.uint8)
            #         # 显示图像
            #         plt.imshow(mask_data, cmap='gray')
            #         plt.axis('off')
            #         plt.suptitle('result_masks')
            #         plt.show()
            # ************************************************
            mask = result_masks_list[:]
            del result_masks_list
            del shen_masks

            target_mask, valid = nested_tensor_from_tensor_list(mask).decompose()
            # Tensor[1, 4, 416, 576]   Tensor[1, 416, 576].bool
            if target_mask.shape[1] > 0:  # num_obj > 0
                target_mask = target_mask.to(outputs_mask)  # target_mask:[1, 4, 416, 576]  outputs_mask:[1, 100, 52, 72]
                target_mask = F.interpolate(target_mask, size=pixel_feature_size, mode="nearest").bool()  # [1, 4, 52, 72]
            else:
                target_mask_size = [target_mask.shape[0], target_mask.shape[1], *pixel_feature_size]
                target_mask = torch.zeros(size=target_mask_size, device=outputs_mask.device).bool()

            gt_attn_mask_size = [   # [2, 108, 56, 80]  [bs, num_query + num_aux_query, h, w]
                outputs_mask.shape[0], self.num_queries + self.num_aux_queries, *pixel_feature_size
            ]
            gt_attn_mask = torch.zeros(size=gt_attn_mask_size, device=outputs_mask.device).bool()  # [2, 108, 56, 80]
            gt_attn_mask[src_idx] = ~ target_mask[tgt_idx]
            gt_attn_mask = gt_attn_mask.flatten(2)  # [2, 108, 4480]

            invalid_gt_query = gt_attn_mask.all(-1, keepdim=True)  # b, n, 1  [1, 108, 1]
            gt_attn_mask = (~invalid_gt_query) & gt_attn_mask  # b, n, h*w  [1, 108, 3744]
            gt_attn_mask = gt_attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)  # [8, 108, 3744]
            gt_attn_mask = gt_attn_mask.detach()
        else:
            matching_indices = None
            gt_attn_mask = None

        return outputs_class, outputs_mask, attn_mask, matching_indices, gt_attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, output_indices, output_query_locations):
        return [
            {
                "query_locations": output_query_locations,
                "pred_logits": a,
                "pred_masks": b,
                "pred_matching_indices": c}
            for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], output_indices[:-1])
        ]
