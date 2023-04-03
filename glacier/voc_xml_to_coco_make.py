"""
@FileName：gadl_read.py\n
@Description：将voc的xml转换为coco的json\n
@Author：Wang.Lei\n
@Time：2023/3/29 16:57\n
@Department：Postgrate\n
"""
import xml.etree.ElementTree as ET
import json
import datetime
from pathlib import Path
import random

coco = dict()
coco['info'] = []
coco['images'] = []
coco['annotations'] = []
coco['categories'] = []
coco['license'] = [{'name' : '测试数据'}]
CategorySet = dict()
bndbox_id = 0
category_item_id = 1


def addInfoItem():
    InfoItem = dict()
    InfoItem['description'] = '测试数据'
    InfoItem['Version'] = 0.1
    coco['info'].append(InfoItem)


def addImageItem(file_name,size):
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id = int(Path(file_name).stem)
    ImageItem = dict()
    ImageItem['id'] = image_id
    ImageItem['date_captured'] = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    ImageItem['file_name'] = file_name
    ImageItem['width'] = size['width']
    ImageItem['height'] = size['height']
    coco['images'].append(ImageItem)

def addCategoryItem(name):
    """

    Args:
        name: 类名

    Returns:

    """
    global category_item_id
    CategoryItem = dict()
    CategoryItem['supercategory'] = 'ice'
    if name not in CategorySet.keys():
        CategorySet[f'{name}'] = category_item_id
        category_item_id += 1
        CategoryItem['name'] = name
        CategoryItem['id'] = CategorySet[f'{name}']
        coco['categories'].append(CategoryItem)
    else:
        CategoryItem['id'] = CategorySet[f'{name}']
    return CategoryItem['id']


def addAnnItem(image_id,category_id,bbox,object_id):
    AnnItem = dict()
    AnnItem['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] - bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] - bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])
    AnnItem['segmentation'].append(seg)
    AnnItem['area'] = bbox[2] * bbox[3]
    AnnItem['iscrowd'] = 0
    AnnItem['ignore'] = 0
    AnnItem['id'] = object_id
    AnnItem['image_id'] = image_id
    AnnItem['category_id'] = category_id
    AnnItem['bbox'] = bbox
    coco['annotations'].append(AnnItem)
    object_id += 1
    return object_id




object_id = 0
def coco_xml(img_list):
    global object_id
    for path in img_list:
        bndbox = dict()
        size = dict()
        tree = ET.parse(path)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag   # 现有节点名
            if current_parent == 'filename':
                file_name = elem.text
            if current_parent == 'size':
                for subelem in elem:
                    size[subelem.tag] = int(subelem.text)
                addImageItem(file_name,size)
            if current_parent == 'object':
                # add img item only after parse <size> tag
                tag = []
                for subelem in elem:
                    tag.append(subelem.tag)
                banbox_count = tag.count('bndbox')
                for subelem in elem:
                    bndbox['xmin'] = None
                    bndbox['xmax'] = None
                    bndbox['ymin'] = None
                    bndbox['ymax'] = None

                    current_sub = subelem.tag       # str
                    if current_sub == 'name':
                        category_id = subelem.text  # 类名
                        CategoryId = addCategoryItem(category_id)
                    if current_sub == 'bndbox':
                        # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                        for i in range(banbox_count):
                            bnd_out=[]
                            for option in subelem[i * 4:i * 4 + 4]:
                                bndbox[option.tag] = float(option.text)
                            bnd_out.append(bndbox['xmin'])
                            bnd_out.append(bndbox['ymin'])
                            bnd_out.append(bndbox['xmax'] - bndbox['xmin'])
                            bnd_out.append(bndbox['ymax'] - bndbox['ymin'])
                            if bndbox[option.tag] is None:
                                raise Exception('xml structure lack bbox')
                            image_id = int(Path(file_name).stem)
                            object_id = addAnnItem(image_id, CategoryId,bnd_out,object_id)
                            object_id = i  + object_id
            # elif current_parent == 'size':
            #     for subelem in elem:
            #         if size[subelem.tag] is not None:
            #             raise Exception('xml structure broken at size tag.')
            #         size[subelem.tag] = int(subelem.text)
            #     addImageItem(file_name,size)
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





if __name__ == '__main__':
	#修改这里的两个地址，一个是xml文件的父目录；一个是生成的json文件的绝对路径
    xml_path = r'H:\daochu_classfied\3channel_tif_voc\Annotations'
    # xml_path = r'H:\daochu_classfied\3channel_tif_voc\coco\ann'
    img_list = list(Path(xml_path).rglob('*.xml'))
    img = {}
    path = random_img(img_list,0.7,0.2)
    img['train'] = path[0]
    img['test'] = path[1]
    img['val'] = path[2]
    for i,j in img.items():
        json_file = r'H:\daochu_classfied\{}.json'.format(i)
        coco_xml(j)
        # json.dump(coco,open(json_file, 'w'))
        with open(json_file, 'w') as output_json_file:
            json_str = json.dumps(coco, indent=1, sort_keys=True)
            output_json_file.write(json_str)
        coco['info'] = []
        coco['images'] = []
        coco['annotations'] = []