import numpy as np
import torch
from pycocotools.coco import COCO   # COCO API
import skimage.io as io   # 加载图片
import matplotlib.pyplot as plt  # 显示图片
import cv2
from torch.nn import functional as F


# 生成mask图
def mask_generator(coco, width, height, anns_list):
    mask_pic = np.zeros((height, width))
    # 生成mask - 此处生成的是4通道的mask图,如果使用要改成三通道,可以将下面的注释解除,或者在使用图片时搜相关的程序改为三通道
    for single in anns_list:
        mask_single = coco.annToMask(single)
        mask_pic += mask_single
    # 转化为255
    for row in range(height):
        for col in range(width):
            if (mask_pic[row][col] > 0):
                mask_pic[row][col] = 255
    mask_pic = mask_pic.astype(int)
    return mask_pic


def mask_augmentation(shen_mask, Aug_Flag):
    kernel = np.ones((3, 3), dtype=np.uint8)
    # 对每个目标的掩码进行膨胀处理
    iteration = np.random.randint(2, 12)
    iteration2 = 2
    iteration2 = 2
    if Aug_Flag == False:
        result_mask = shen_mask
    else:
        dilat = cv2.dilate(shen_mask, kernel, iterations=iteration)
        result_mask = cv2.erode(shen_mask, kernel, iterations=iteration2)
        # result_mask = dilat - result_mask
    return result_mask


mode = 1  # 1 原图   2 制作真值   3 mask映射
save = 1  # 1 保存
Aug_Flag = False
dataset_mode = 'train'   #  val / train
annFile = 'datasets/coco/annotations/instances_' + dataset_mode +'2017.json' # 指定标注文件的标注路径
coco = COCO(annFile)  # 通过文件路径加载COCO数据集

for ids in coco.imgs.keys():
# for ids in [184613, 328757, 509822, 321107, 394892, 310391, 310103, 250108, 354533, 450263, 207797, 511058, 545959,
#             554348, 522778, 571034]:
    ids = 89668
    print("ids:", ids)
    picture_name = str(ids)
    img = coco.loadImgs([ids])[0]  # 图像的所有信息
    if dataset_mode == 'val':
        I = io.imread('datasets/coco/val2017/' + img['file_name'])  # 图像本身
    else:
        I = io.imread('datasets/coco/train2017/' + img['file_name'])  # 图像本身
    I = I.copy()
    h, w = I.shape[0], I.shape[1]

    # ****************************************************************
    if mode == 2 or mode == 1:
        if mode == 2:
            zero_I = np.zeros_like(I)
            I = zero_I
        plt.imshow(I)
        annIds = coco.getAnnIds(imgIds=img['id'])  # 对应图像的所有标注索引
        anns=coco.loadAnns(annIds)  # 对应图像的所有标注
        if mode == 1:
            coco.showAnns(anns)
        else:
            for i in range(anns.__len__()):
                if i == 1:
                    break
                single_ann = [anns[i]]
                coco.showAnns(single_ann)
                print("第", i, "个实例")

        # 调整coco.py文件

    if mode == 3:
        # 读取mask
        # ****************************************************************
        mask = io.imread("output/gt_image/" + picture_name + '.png')
        mask = np.array(mask.transpose(2, 0, 1)[0]).astype(np.float32)
        mask = mask_augmentation(mask, Aug_Flag)# .astype(np.bool)
        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=[h, w], mode="bilinear")
        mask = mask.squeeze(0).squeeze(0)
        mask = np.array(mask, dtype=bool).astype(int)
        # ****************************************************************

        mask_and_image = I
        mask_and_image[:, :, 0] = np.multiply(mask, I[:, :, 0])
        mask_and_image[:, :, 1] = np.multiply(mask, I[:, :, 1])
        mask_and_image[:, :, 2] = np.multiply(mask, I[:, :, 2])
        plt.imshow(mask_and_image)

    # plt.imshow(mask, cmap='gray')
    plt.axis('off')
    if save == 1:
        plt.savefig(fname='output/gt_image/' + picture_name, bbox_inches='tight', pad_inches=0)
    plt.show()
    flag = input("请输入Y/N: ")
    if flag == "N" or flag == "n":
        break
    continue



