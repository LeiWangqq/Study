"""
@FileName：coco_correct.py\n
@Description：\n
@Author：Wang.Lei\n
@Time：2022/11/29 15:31\n
@Department：Postgrate\n
"""

import json
from pathlib import Path
import numpy as np
from mmdetection_learning import pycococreatortools
from skimage import io
from skimage import measure
import imageio


def find_files(path, endwith='tif'):
    """
    遍历文件夹，找到所需相应后缀名所有文件
    Args:
        path: 文件夹路径
        endwith: 后缀名，需要带上 .

    Returns:
    由winpath格式组成的相应后缀名文件
    """
    initial = []
    find_function = lambda path, endwith: Path(path).rglob('*' + endwith)
    path_generator = find_function(path, endwith)
    for i in path_generator:
        initial.append(i)
    ans = initial
    return ans

def read_img(img_path):
    """
    读取 tif 文件，返回前三波段图像数组
    Args:
        img_path: 文件父路径

    Returns:
    所有文件组成的列表
    """
    data = imageio.v3.imread(img_path)
    return data[:, :, 0:3]

def create_images_info(image_id, file_name, image_size):
    """
    创建重要信息项目 images
    Args:
        image_id: id数值
        file_name:  文件名
        image_size: 图像长宽

    Returns:
    images项
    """
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
    }

    return image_info


def build_json(ROOT_DIR, IMAGE_DIR, ANNOTATION_DIR):
    """
    根据已分类切片进行coco数据集，制作
    Args:
        ROOT_DIR: 保存根目录
        IMAGE_DIR: 读取images图像路径
        ANNOTATION_DIR: 读取对应images文件的labels文件路径

    Returns:
    满足coco数据集格式的弓网文件
    """

    # 定义coco数据集的相关数据项，【以键对形式储存】
    INFO = {}
    LICENSE = {}
    CATEGORIES = [
        {
            'id': 1,
            'name': 'gongwang',
            'supercategory': 'gongwang',
        },
    ]
    coco_output = {
        "info": INFO,
        "licenses": {},
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    annotation_id = 0

    file_list = find_files(IMAGE_DIR, '.tif')  # 形成图像路径的列表

    for image_filename in file_list:  # 遍历每张图像
        image = read_img(str(image_filename))
        image_id = int(image_filename.name.rstrip(image_filename.suffix))
        image_size = image.shape  # 读取为(H,W,C)，这里会影响json文件中的"width"和"height"
        image_info = create_images_info(
            image_id, image_filename.name, image_size[0:2])
        coco_output["images"].append(image_info)


        annotation_filename = ANNOTATION_DIR / Path(image_filename.name)
        # 遍历图像的ground truth，这样确保了"annotation"和"image"中的"id"对应

        category_info = {'id': 1, 'is_crowd': 0}  # 数据集只有１类，其值为classvalue,且'is_crowd': 0,多边形顶点表示

        # binary_mask = np.asarray(io.imread(annotation_filename).astype(np.uint8))
        binary_image = imageio.v3.imread(annotation_filename)  # 读为uint16

        if binary_image.any() > 0:  # 如果二值掩膜中有元素大于0，即存在1
            labels = measure.label(binary_image, connectivity=2)  # 标记连通域，2表示8联通，不同连通域用1，2，3等做标记
            for i in range(1, labels.max() + 1):  # 遍历每个连通域

                # labels_slice = (labels == i)：返回一个bool矩阵，为i的地方是True,其它地方为False
                labels_slice = (labels == i).astype('uint8') * 255  # 分离出ground truth中的每个连通域

                annotation_id = annotation_id + 1

                annotation_info = pycococreatortools.create_annotation_info(
                    annotation_id=annotation_id,
                    image_id=image_id,
                    category_info=category_info,
                    binary_mask=labels_slice,
                    tolerance=0,
                    image_size=None,
                )


                coco_output["annotations"].append(annotation_info)


    final_name = Path(IMAGE_DIR).name  # train,test or validation
    with open(f'{ROOT_DIR}/{final_name}.json', 'w') as output_json_file:
        json_str = json.dumps(coco_output, indent=2, sort_keys=True, ensure_ascii=False)
        output_json_file.write(json_str)


if __name__ == "__main__":
    # 具体怎么传入看自己的文件夹结构
    ROOT_DIR = r'H:\daochu_classfied'
    IMAGE_DIR = ROOT_DIR + r'\images'
    ANNOTATION_DIR = ROOT_DIR + r'\labels'

    build_json(ROOT_DIR, IMAGE_DIR, ANNOTATION_DIR)