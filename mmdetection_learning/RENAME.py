"""
@FileName：RENAME.py\n
@Description：\n
@Author：Wang.Lei\n
@Time：2022/11/25 18:57\n
@Department：Postgrate\n
"""
from pathlib import Path
import shutil

from tqdm import trange


# import os
# filter_tif = lambda image_dir,endwith:sorted([os.path.join(image_dir,file) for file in os.listdir(image_dir) if file.endswith(endwith)])

def find_files(path, endwith = 'tif'):
    initial = []
    find_function = lambda path, endwith: Path(path).rglob('*' + endwith)
    path_generator = find_function(path, endwith)
    for i in path_generator:
        initial.append(i)
    ans = initial
    return ans

def save_style_length(filename,adds=0,endwith='.tif'):
    """
    对原文件增加数值后，保存文件名的长度，如000000000.tif 而不是0.tif
    Args:
        filename: 输入文件名
        adds: 需要加上的数值
        endwith: 后缀名，默认 .tif

    Returns:
        保存长度格式的文件名
    """
    old_name_num = filename.name.rstrip(f'{endwith}')
    old_name_num_length = len(old_name_num)  # 命名的名称长度
    new_name_num = int(old_name_num)+adds
    new_name_num_length = len(f'{new_name_num}')

    if new_name_num_length < old_name_num_length:
        new_name_nums = (old_name_num_length-new_name_num_length)*'0'+f'{new_name_num}'+endwith
    else:
        new_name_nums = f'{new_name_num}'+endwith
    return new_name_nums
images_name=Path('E:/Desktop/DataSet_new/北京东郊/images/00001.tif')
save_style_length(images_name,0)




def rename(images_path_1,images_path_2,save_path,endwith='.tif',save_style=False):
    """
    # 仅仅更改为int整数形式，不保留原有命名格式，如000000000.tif
    Args:
        images_path_1（str）: # 原数据集images文件父文件夹
        images_path_2(str) # 增加数据集images文件父文件夹
        save_path: 保存父路径名 ，保存在 save_path+ images/labels下
        endwith : 默认 .tif ，可改后缀,注意小数点
        save_style：是否保存图片原文件名格式

    Returns:
        重命名后的图像文件夹 save_path+ images/labels
    """
    image_generate_path = Path(save_path) / 'images'
    labels_generate_path = Path(save_path) / 'labels'
    Path(image_generate_path).mkdir(parents=True,exist_ok=True)
    Path(labels_generate_path).mkdir(parents=True, exist_ok=True)


    images_list_1 = find_files(images_path_1)
    labels_list_1 = find_files(str(find_files(images_path_1)[0].parent).rstrip('images') + 'labels')
    images_list_2 = find_files(images_path_2)
    labels_list_2 = find_files(str(find_files(images_path_2)[0].parent).rstrip('images')+'labels')
    origin_num = len(images_list_1)  # 第一数据集文件个数
    second_num = len(images_list_2) # 第二数据集文件个数


    if save_style == False:
        for i in trange(origin_num):
            images_name = images_list_1[i]
            labels_name = labels_list_1[i]

    # winpath 与 str 组成路径只需 /
            shutil.copy(images_name, image_generate_path / f'{str(i)+endwith}')
            shutil.copy(labels_name, labels_generate_path / f'{str(i)+endwith}')
        for j in trange(second_num):
            images_name = images_list_2[j]
            labels_name = labels_list_2[j]
            shutil.copy(images_name,image_generate_path / (f'{str(j+origin_num)}'+images_name.suffix))
            shutil.copy(labels_name, labels_generate_path / (f'{str(j+origin_num)}'+images_name.suffix))
    else:
        for i in trange(origin_num):
            images_name = images_list_1[i]
            labels_name = labels_list_1[i]

    # str 与 str 组成路径中间需要r'\\'
            shutil.copy(images_name, str(image_generate_path)+r'\\'+save_style_length(images_name,-1))
            shutil.copy(labels_name, str(labels_generate_path)+r'\\'+save_style_length(labels_name,-1))
        for j in trange(second_num):
            images_name = images_list_2[j]
            labels_name = labels_list_2[j]
            shutil.copy(images_name, str(image_generate_path)+r'\\'+save_style_length(images_name,origin_num-1))
            shutil.copy(labels_name, str(labels_generate_path)+r'\\'+save_style_length(labels_name,origin_num-1))

#
if __name__ == '__main__':
    images_path_1 = r'E:\Desktop\DataSet_new\北京东郊\images'
    images_path_2 = r'E:\Desktop\DataSet_new\济南京沪\images'  # 增加数据集读取images图像路径
    save_path = r'E:\Desktop\DataSet_new\Rename'
    save_path_save_style = r'E:\Desktop\DataSet_new\Rename_save_name_style'      # 保存主路径，会保存在 save_path+ images/labels下

    rename(images_path_1,images_path_2,save_path,save_style=False)
    rename(images_path_1, images_path_2, save_path_save_style, save_style=True)

