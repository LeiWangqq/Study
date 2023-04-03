"""
@FileName：voc_make.py\n
@Description：将深度学习导出的文件取相应通道，并划分train、test、val数据集\n
@Author：Wang.Lei\n
@Time：2023/3/24 15:39\n
@Department：Postgrate\n
"""

import imageio
import numpy as np
from pathlib import Path
import os
import shutil
import random
from tqdm import trange

def image_io(img_path, name_path):
    # 保证不对数据做任何变换，读取进出均为 uin16
    # imageio 读取进来为 HWC
    img = imageio.v3.imread(img_path)
    # 若用imageio.v3.imwrite 需要输入 CHW，【注意】此处需要转换
    img_band123 = np.stack([img[:, :, 3],img[:, :, 2],img[:, :, 1]],axis = 2)
    img_band123=((img_band123-img_band123.min())/(img_band123.max()-img_band123.min())*255).astype('uint8')
    imageio.imsave(name_path,img_band123)

def find_files(path, endwith='tif'):
    initial = []
    find_function = lambda path, endwith: Path(path).rglob('*' + endwith)
    path_generator = find_function(path, endwith)
    for i in path_generator:
        initial.append(i)
    ans = initial
    return ans
def dataset_slice(image_dir,labels_dir,save_path,train_per,val_per):
    #  创建目录
    JPEGImages_dir = save_path + r'\JPEGImages'
    ImageSets_Main_dir = save_path + r'\ImageSets\Main'
    Annoation_dir = save_path + r'\Annotations'

    # 如果不存在，则建立该文件夹
    Path(JPEGImages_dir).mkdir(exist_ok=True, parents=True)
    Path(ImageSets_Main_dir).mkdir(exist_ok=True, parents=True)
    Path(Annoation_dir).mkdir(exist_ok=True, parents=True)


    images_list = find_files(image_dir,'.tif')
    labels_list = find_files(labels_dir,'.xml')

    # 不同样本量
    num = len(images_list)
    t_r = int(num * train_per)
    t_v = int(num * val_per)
    # 随机抽样，获取索引值
    train = random.sample(images_list, t_r)
    val = random.sample(images_list, t_v)

    # 创建txt文件
    file_train_val = open(ImageSets_Main_dir + '/trainval.txt', 'w')
    file_train = open(ImageSets_Main_dir + '/train.txt', 'w')
    file_val = open(ImageSets_Main_dir + '/val.txt', 'w')
    file_test = open(ImageSets_Main_dir + '/test.txt', 'w')


    for i in trange(num):
        img_dir = images_list[i]
        if len(find_files(JPEGImages_dir))==num:
            image_io(img_dir, JPEGImages_dir + r'\\' + images_list[i].name)
            # shutil.copy(labels_list[i],Annoation_dir+r'\\'+labels_list[i].name)
        # name = images_list[i].stem+'\n'
        # if img_dir in train:
        #     file_train_val.write(name)
        #     file_train.write(name)
        # elif img_dir in val:
        #     file_train_val.write(name)
        #     file_val.write(name)
        # else:
        #     file_test.write(name)


if __name__ == '__main__':
    save_path = r'H:\daochu_classfied\3channel_tif_voc'
    img_dir = r'H:\daochu_classfied\images'
    label_dir = r'H:\daochu_classfied\labels'
    dataset_slice(img_dir, label_dir, save_path, 0.7, 0.2)