from pycocotools.coco import COCO   # COCO API
import skimage.io as io   # 加载图片
import matplotlib.pyplot as plt  # 显示图片


def get_gt_szie(image, img_info):
    height, width = img_info['height'], img_info['width']
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    top_x = int(center_x - width / 2)
    top_y = int(center_y - height / 2)
    bottle_x = top_x + width
    bottle_y = top_y + height
    return image[top_y:bottle_y + 1, top_x:bottle_x + 1, :]


def visual(ids):
    img = coco.loadImgs([ids])[0]  # 图像的所有信息
    height, width = img['height'], img['width']
    I = io.imread(image_dir_path + img['file_name'])  # 图像本身
    # I = get_gt_szie(I, img)
    plt.imshow(I)
    plt.axis("off")
    annIds = coco.getAnnIds(imgIds=img['id'])  # 对应图像的所有标注索引
    anns=coco.loadAnns(annIds)  # 对应图像的所有标注
    coco.showAnns(anns)
    plt.savefig(fname="/data3/shenbaoyue/code/FastInst/Fastinst-SQR-select/output/gt_image/" + str(img['id']) + ".jpg", bbox_inches='tight', pad_inches=0)
    plt.show()
    # 清除图形窗口
    plt.clf()  # 或者 plt.close()


if __name__ == "__main__":
    mode = "train"
    annFile = None
    image_dir_path = None

    if mode == "val":
        annFile = '/data3/shenbaoyue/code/FastInst/Fastinst-SQR-select/datasets/coco/annotations/instances_val2017.json'
        image_dir_path = '/data3/shenbaoyue/code/FastInst/Fastinst-SQR-select/datasets/coco/val2017/'
    else:
        annFile = '/data3/shenbaoyue/code/FastInst/Fastinst-SQR-select/datasets/coco/annotations/instances_train2017.json'  # 指定标注文件的标注路径
        image_dir_path = '/data3/shenbaoyue/code/FastInst/Fastinst-SQR-select/datasets/coco/train2017/'
    coco = COCO(annFile)  # 通过文件路径加载COCO数据集

    index = 0
    for ids in coco.imgs.keys():
        # ids = 36
        visual(ids)
        # break
        if ids % 20 == 0: 
            flag = input("请输入Y/N: ")
            if flag == "N" or flag == "n":
                break
            continue
        index += 1



