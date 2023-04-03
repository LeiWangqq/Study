#!/usr/bin/env python3


import re
import numpy as np
from itertools import groupby
from skimage import measure
from PIL import Image
from pycocotools import mask


convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):      # 只有形状和元素全部相同才能返回True,为False的时候这句话执行，即两array不相等，即这条轮廓线不闭合，头尾不一致
        contour = np.vstack((contour, contour[0]))       # 按垂直顺序(行顺序)叠加，叠加成一个新的array，使它头尾一致，与之相对：np.hstack()
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0) # 扩展是为了保证轮廓不会超出整张图的范围
    contours = measure.find_contours(padded_binary_mask, 0.5)                                 # 找到轮廓点，shape:由多个n*2的数组组成的列表,2代表行列，每个n*2数组代表一条轮廓线，这个数组的元素就是每个像素点
    contours = np.subtract(contours, 1)                                                       # 所有坐标减去1，还原回原来没有pad的坐标位置
    for contour in contours:                                                                  # 遍历每条轮廓线
        contour = close_contour(contour)                                                      # 使轮廓线头尾像素的行列号相同，使之闭合
        contour = measure.approximate_polygon(contour, tolerance)                             # 用指定的公差即tolerance拟合多边形的边，公差为0，则返回原来的行列号
        if len(contour) < 3:                                                                  # 该多边形边的组成长度
            continue
        contour = np.flip(contour, axis=1)                                                    # 转置维度,这里交换行号和列号，为什么要交换：因为要以xy坐标来写入，即列行
        segmentation = contour.ravel().tolist()                                               # 展平，然后转为列表,展平是浅复制，尽量避免用这种，以flatten()替代
        # after padding and subtracting 1 we may get -0.5 points in our segmentation          # 可能有-0.5的行列号
        segmentation = [0 if i < 0 else i for i in segmentation]                              # 将-0.5置为0
        polygons.append(segmentation)  # 所有数组成一个列表，数按照相邻的顺序两两组成一个点的xy坐标
    return polygons


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))
    return rle



def create_image_info(image_id, file_name, image_size):
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
    }

    return image_info

def create_annotation_info(annotation_id, image_id, category_info, binary_mask, 
                           image_size=None, tolerance=2, bounding_box=None):

    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask, image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))  # np.asfortranarray:转为array,并使array在内存中按fortran的顺序排列
    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    if category_info["is_crowd"]:
        is_crowd = 1
        segmentation = binary_mask_to_rle(binary_mask)
    else :
        is_crowd = 0
        segmentation = binary_mask_to_polygon(binary_mask, tolerance)
        if not segmentation:
            return None

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": is_crowd,
        "area": area.astype(float).tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,             # [[]]
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    }
    return annotation_info
