"""
@FileName：spilt_train_val.py\n
@Description：获取图像并将其划分为训练集、测试集、验证集\n
@Author：Wang.Lei\n
@Time：2022/11/18 13:32\n
@Department：Postgrate\n
"""
import random
from pathlib import Path
import argparse
from mmdetection_learning.gadl_read import find_files
from tqdm import trange


# 输入根目录
path = Path(r'E:\Desktop\数据\北京东郊')

# 调用 argparse 并将其转化为对象
parser = argparse.ArgumentParser()
# 【注意】 add_argument中   --表示非必要参数，若无此标识符，则为必要参数
parser.add_argument('--img_path',default= path / 'images',type=str,help='input img path')
parser.add_argument('--txt_path',default= str(path / 'dataSet'),type=str,help='output txt path')
opt = parser.parse_args()


# 设置比例值
trainval_per = 0.8
train_per = 0.72
img_open_path = opt.img_path
txt_save_path = opt.txt_path
total_xml = find_files(path/'jpg','jpg')
# total=[]
# for i in total_xml:
#     total.append(str(i).rstrip('.jpg').lstrip('E:\\Desktop\\222\\Pascal\\jpg'))
Path(path / 'dataSet').mkdir(exist_ok=True,parents=True)     # exist_ok = True 存在时才不会报错

num = len(total_xml)
list_index = range(num)
t_v = int(num*trainval_per)
t_r = int(num*train_per)
# 随机抽样，获取索引值
train_val = random.sample(list_index , t_v)
train = random.sample(list_index , t_r)


file_train_val = open(txt_save_path + '/trainval.txt', 'w')
file_train = open(txt_save_path + '/train.txt', 'w')
file_val = open(txt_save_path + '/val.txt', 'w')
file_test = open(txt_save_path + '/test.txt', 'w')

for i in trange(num):
    name = str(total_xml[i]).lstrip(r'E:\\Desktop\\222\\Pascal\\jpg').rstrip('.jpg')+'\n'
    if i in train_val:
        file_train_val.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)
    else:
        file_test.write(name)
file_train_val.close()
file_train.close()
file_val.close()
file_test.close()
