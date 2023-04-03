"""
@FileName：spilt_recorrect.py\n
@Description：\n
@Author：Wang.Lei\n
@Time：2022/11/25 13:34\n
@Department：Postgrate\n
"""
import os
import shutil
from pathlib import Path
from tqdm import trange
import random
from mmdetection_learning.coco import build_json




# 划分数据集,label也要一起划分，划分完后时候building_to_coco.py 脚本输出json文件
filter_tif = lambda image_dir,endwith:sorted([os.path.join(image_dir,file) for file in os.listdir(image_dir) if file.endswith(endwith)])
def dataset_slice(image_dir,labels_dir,save_path,train_per,val_per,test_per):

    #  创建目录
    train_images_save_path = save_path + r'\images\train'
    test_images_save_path = save_path + r'\images\test'
    validation_images_save_path = save_path + r'\images\validation'
    train_labels_save_path = save_path + r'\labels\train'
    test_labels_save_path = save_path + r'\labels\test'
    validation_labels_save_path = save_path + r'\labels\validation'

    Annoation = save_path + r'\Annoation'
    # json_train = save_path+r'/Annoation/train'
    # json_test = save_path + r'/Annoation/test'
    # json_val = save_path + r'/Annoation/val'

    # 如果不存在，则建立该文件夹
    Path(train_images_save_path).mkdir(exist_ok=True, parents=True)
    Path(test_images_save_path).mkdir(exist_ok=True, parents=True)
    Path(validation_images_save_path).mkdir(exist_ok=True, parents=True)
    Path(train_labels_save_path).mkdir(exist_ok=True, parents=True)
    Path(test_labels_save_path).mkdir(exist_ok=True, parents=True)
    Path(validation_labels_save_path).mkdir(exist_ok=True, parents=True)

    # Path(json_train).mkdir(exist_ok=True, parents=True)
    # Path(json_test).mkdir(exist_ok=True, parents=True)
    # Path(json_val).mkdir(exist_ok=True, parents=True)
    Path(Annoation).mkdir(exist_ok=True, parents=True)


    image_list = filter_tif(image_dir,'.tif')
    labels_list = filter_tif(labels_dir,'.tif')

    # 不同样本量
    num = len(image_list)
    list_index = range(num)
    t_r = int(num * train_per)
    t_v = int(num * val_per)
    # 随机抽样，获取索引值
    train = random.sample(list_index, t_r)
    val = random.sample(list_index, t_v)
    j=k=l=0


    for i in trange(num):
        images_name = image_list[i]
        labels_name = labels_list[i]

        if i in train:
            shutil.copy(images_name, train_images_save_path+f'\{j}.tif')
            shutil.copy(labels_name, train_labels_save_path+f'\{j}.tif')
            j+=1
        elif i in val:
            shutil.copy(images_name,validation_images_save_path+f'\{k}.tif')
            shutil.copy(labels_name, validation_labels_save_path+f'\{k}.tif')
            k+=1
        else:
            shutil.copy(images_name,test_images_save_path+f'\{l}.tif')
            shutil.copy(labels_name, test_labels_save_path+f'\{l}.tif')
            l+=1
    build_json(Annoation,train_images_save_path,train_labels_save_path)
    build_json(Annoation, validation_images_save_path, validation_labels_save_path)
    build_json(Annoation, test_images_save_path, test_labels_save_path)


if __name__ == '__main__':
    save_path = r'H:\daochu_classfied'   # 保存父目录
    image_dir = save_path+r'\images'        # 读取三通道图像目录
    labels_dir = save_path + r'\labels'    # 读取图像对应label目录
    dataset_slice(image_dir,labels_dir,save_path,0.8,0.2,0)   # 后三位为train、val、test
