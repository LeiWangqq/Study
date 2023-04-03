"""
@FileName：Mask_to_Json.py\n
@Description：将掩膜文件转换为coco格式\n
@Author：Wang.Lei\n
@Time：2023/4/3 10:30\n
@Department：Postgrate\n
"""

import datetime
import json
import os
import random
import pycococreatortools
from matplotlib import Path
from study.mmdetection_learning.gadl_read import readTiff


INFO = {
    "description": "Leaf Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2017,
    "contributor": "Francis_Liu",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

# 根据自己的需要添加种类
CATEGORIES = [
    {
        'id': 1,
        'name': 'glacier lake',
        'supercategory': 'ice',
    }
]


def find_files(path, endwith='tif'):
    initial = []
    find_function = lambda path, endwith: Path(path).rglob('*' + endwith)
    path_generator = find_function(path, endwith)
    for i in path_generator:
        initial.append(str(i))
    ans = initial
    return ans



def main(ROOT_DIR,image_files,ANNOTATION_DIR):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    segmentation_id = 0

    # filter for select images，here is tif
    # image_files = find_files(IMAGE_DIR)

    # go through each image
    for image_filename in image_files:
        # image = Image.open(image_filename)
        size = readTiff(image_filename)[0].shape[1:]
        image_id = int(Path(image_filename).stem)
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_filename), size)
        coco_output["images"].append(image_info)


    # # filter for associated png annotations
    # annotation_files = find_files(ANNOTATION_DIR)
    # for annotation_filename in annotation_files:

        # go through each associated annotation
        annotation_filename = ANNOTATION_DIR + r'\\' + Path(image_filename).name
        print(annotation_filename)
        # class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename]
        class_id = 1
        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}

        # mask需要为二值化图像
        # binary_mask = np.asarray(Image.open(annotation_filename)
        #                          .convert('1')).astype(np.uint8)
        binary_mask = readTiff(annotation_filename)[0]


        annotation_info = pycococreatortools.create_annotation_info(
            segmentation_id, image_id, category_info, binary_mask,
            size, tolerance=2)
        print(image_id,segmentation_id)
        if annotation_info is not None:
            coco_output["annotations"].append(annotation_info)

        segmentation_id = segmentation_id + 1


    with open('{}.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


def random_img(img_list,train,test):
    '''

    Args:
        img_list: 输入xml的文件列表，windowpath格式的
        train: 训练集占比
        test: 测试集占比

    Returns:训练集、测试集和、验证集列表

    '''
    random.shuffle(img_list)
    train_len = int(len(img_list)*train)
    test_len = int(len(img_list)*test)
    val_len = len(img_list)-train_len-test_len
    train = img_list[:train_len]
    test = img_list[train_len:train_len+test_len]
    val = img_list[train_len+test_len:train_len+test_len+val_len+1]
    return train,test,val

if __name__ == "__main__":
    '''
    # 生成总Json文件
    ROOT_DIR = r'H:\daochu_classfied_haved'
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    ANNOTATION_DIR = os.path.join(ROOT_DIR, "labels")
    main(ROOT_DIR,IMAGE_DIR,ANNOTATION_DIR)
    '''
    IMAGES_dir = r'H:\daochu_classfied_haved\images'
    ROOT_DIR = r'H:\daochu_classfied_haved'
    ANNOTATION_DIR = ROOT_DIR + r"\labels"
    img_list = [str(i) for i in list(Path(IMAGES_dir).rglob('*.tif'))]
    # 划分数据集
    img = dict()
    path = random_img(img_list, 0.7, 0.2)
    img['train'] = path[0];img['test'] = path[1];img['val'] = path[2]

    for name,path_value in img.items():
        IMAGE_FILES = path_value

        Name = ROOT_DIR+'\\'+name
        main(Name, IMAGE_FILES, ANNOTATION_DIR,)
