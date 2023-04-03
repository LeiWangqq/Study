import numpy as np
import os
import shutil
from tqdm import trange

def mkdir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        pass

# 划分数据集,label也要一起划分，划分完后时候building_to_coco.py 脚本输出json文件
filter_tif = lambda image_dir,endwith:sorted([os.path.join(image_dir,file) for file in os.listdir(image_dir) if file.endswith(endwith)])
def dataset_slice(image_dir,save_path):
    # 7：2：1切分数据集,比例可以改
    proportion = np.array([0.70,0.20,0.10])  # 相加必须等于1

    train_images_save_path = save_path + r'\train'
    test_images_save_path = save_path + r'\test'
    validation_images_save_path = save_path + r'\validation'

    # 如果不存在，则建立该文件夹
    mkdir(train_images_save_path)
    mkdir(test_images_save_path)
    mkdir(validation_images_save_path)


    image_list = filter_tif(image_dir,'.tif')

    percent_10 = round(len(image_list) / 10)  # 把整个数据集分成10份
    train_images_list = image_list[:int(proportion[0] * 10) * percent_10]  # 训练集占前面7份
    test_images_list = image_list[int(proportion[0] * 10) * percent_10:int(proportion[0] * 10) * percent_10 + int(proportion[1] * 10) * percent_10] # 2份
    validation_images_list = image_list[int(proportion[0] * 10) * percent_10 + int(proportion[1] * 10)  * percent_10:]                          # 最后一份

    # 如果训练集数目不能被2整除
    if len(train_images_list) % 2 != 0:
        train_images_list.pop()

    for i in train_images_list:
        assert int(os.path.splitext(os.path.basename(train_images_list[0]))[0]) == 0  # 判断第一张训练图像的id是否为0
        shutil.copy(i, f'{train_images_save_path}\{int(os.path.splitext(os.path.basename(i))[0])}.tif')

    for i in test_images_list:
        start_num = int(os.path.splitext(os.path.basename(test_images_list[0]))[0])
        shutil.copy(i,f'{test_images_save_path}\{int(os.path.splitext(os.path.basename(i))[0]) - start_num}.tif')

    for i in validation_images_list:
        start_num = int(os.path.splitext(os.path.basename(validation_images_list[0]))[0])
        shutil.copy(i,f'{validation_images_save_path}\{int(os.path.splitext(os.path.basename(i))[0]) - start_num}.tif')


save_path = r'E:\Desktop\jnjh_coco\save_labels'
image_dir =r'E:\Desktop\数据\济南京沪\images'
dataset_slice(image_dir,save_path)