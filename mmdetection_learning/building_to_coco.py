#!/usr/bin/env python3
import json
import os
import numpy as np
import pycococreatortools
# import rasterio
from skimage import io
from skimage import measure
import imageio
import pathlib


def find_files(path, endwith='tif'):
    initial = []
    find_function = lambda path, endwith: pathlib.Path(path).rglob('*' + endwith)
    path_generator = find_function(path, endwith)
    for i in path_generator:
        initial.append(i)
    ans = initial
    return ans

# 将一个文件夹中的tif文件全提出来，形成一个列表
# filter_tif = lambda image_dir,endwith:sorted([os.path.join(image_dir,file) for file in os.listdir(image_dir) if file.endswith(endwith),key=lambda file:])
def read_img(img_path):
    '''读取tiff文件'''
    # dataset = gdal.Open(img_path)
    # width = dataset.RasterXSize  # 栅格矩阵的列数
    # height = dataset.RasterYSize #栅格矩阵的行数
    # bands = dataset.RasterCount #波段数
    # data = dataset.ReadAsArray(0,0,width,height)#获取数据，读为array
    data = imageio.imread(img_path)
    return data[:,:,0:3]


INFO = {}
LICENSE= {}

# 只有一类弓网
CATEGORIES = [
    {
        'id': 1,
        'name': 'gongwang',
        'supercategory': 'gongwang',
    },
]
ROOT_DIR = r'E:\Desktop\DataSet\济南京沪'
IMAGE_DIR =r'E:\Desktop\DataSet\济南京沪\shuffle\images\train'
ANNOTATION_DIR =r'E:\Desktop\DataSet\济南京沪\shuffle\labels\train'

# ROOT_DIR = r'E:\Desktop\数据\北京东郊'
# IMAGE_DIR = r'E:\Desktop\数据\北京东郊\images'
# ANNOTATION_DIR = r'E:\Desktop\数据\北京东郊\labels'

def main(ROOT_DIR,IMAGE_DIR,ANNOTATION_DIR):
    coco_output = {
        "info": INFO,
        "licenses": {},
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    annotation_id=0
    
    file_list = find_files(IMAGE_DIR,'.tif')        # 形成图像路径的列表

    for image_filename in file_list: #遍历每张图像
        image = read_img(image_filename)
        # image=io.imread(IMAGE_DIR+'/'+image_filename)
        # image_id = int(os.path.splitext(os.path.basename(image_filename))[0]) #如文件名xx/xx/xx/xx/957.tif，则只会取出957
        image_id = int(os.path.splitext(os.path.basename(image_filename))[0])

        image_size = image.shape #读取为(H,W,C)，这里会影响json文件中的"width"和"height"

        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_filename), image_size[0:2]) #若是io.imread读取，则为image_size[0:2]
        coco_output["images"].append(image_info)

        annotation_filename = f'{ANNOTATION_DIR}\\{os.path.basename(image_filename)}' #遍历图像的ground truth，这样确保了"annotation"和"image"中的"id"对应
        # print(annotation_filename)

        category_info = {'id': 1, 'is_crowd': 0} #数据集只有１类，其值为classvalue,且'is_crowd': 0
        # binary_mask = np.asarray(io.imread(annotation_filename).astype(np.uint8))
        binary_image = np.asarray(io.imread(annotation_filename))  # 读为uint16

        if binary_image.any() > 0:        # 如果二值掩膜中有元素大于0，即存在1
            labels=measure.label(binary_image, connectivity=2) #标记连通域，2表示8联通，不同连通域用1，2，3等做标记
            for i in range(1, labels.max() + 1):  # 遍历每个连通域

                # labels_slice = (labels == i)：返回一个bool矩阵，为i的地方是True,其它地方为False
                labels_slice = (labels == i).astype('uint8') * 255 #分离出ground truth中的每个连通域

                annotation_id = annotation_id + 1

                annotation_info = pycococreatortools.create_annotation_info(
                    annotation_id = annotation_id,
                    image_id = image_id,
                    category_info = category_info,
                    binary_mask = labels_slice,
                    tolerance = 0,
                    image_size = None,
                )
                # annotation_info = pycococreatortools.create_annotation_info(
                #     annotation_id, image_id, category_info, labels_slice,
                #     image_size[0:2], tolerance=0)

                coco_output["annotations"].append(annotation_info)

            
    with open('{}/train_label.json'.format(ROOT_DIR), 'w') as output_json_file:
        json_str = json.dumps(coco_output, indent = 2, sort_keys = True, ensure_ascii = False)
        output_json_file.write(json_str)

if __name__ == "__main__":
    main(ROOT_DIR,IMAGE_DIR,ANNOTATION_DIR)
