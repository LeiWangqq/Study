"""
@FileName：voc_label.py\n
@Description：生成yolo网络训练所需格式\n
@Author：Wang.Lei\n
@Time：2023/3/27 14:51\n
@Department：Postgrate\n
"""
import os
import xml.etree.ElementTree as ET
from os import getcwd

sets = ['train', 'test', 'val']
# -----------------------------------已知类名----------------------------------------
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
           "tvmonitor"]
# -----------------------------------未知类名----------------------------------------
classes = []
def gen_classes(year, image_id):
    in_file = open('VOCdevkit-car/VOC%s/Annotations/%s.xml' % (year, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        if cls_name in classes:
            pass
        else:
            classes.append(cls_name)
            print("classes name is :", classes)
    return classes




def convert(size, box):
    '''

    Args:
        size: 【width,height】
        box: 【xmin,xmax,ymin,ymax】

    Returns:
        x_middle, y_middle, bbox_width/width, bbox_height/height

    '''
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open('VOCdevkit/VOC/Annotations/%s.xml' % (image_id))
    out_file = open('VOCdevkit/VOC/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):         # 获取所有名为 object 的子节点
        if obj.find('difficult'):
            difficult = obj.find('difficult').text
        else:
            difficult = 0
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:   # 判断是否为难以识别的目标
            continue
        cls_id = classes.index(cls)         # 寻找上方类
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text),
             float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
print(wd)
for image_set in sets:
    if not os.path.exists('VOCdevkit/VOC/labels/'):
        os.makedirs('VOCdevkit/VOC/labels/')
    image_ids = open('VOCdevkit/VOC/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    list_file = open('VOCdevkit/VOC/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('VOCdevkit/VOC/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
