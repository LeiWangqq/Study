"""
@FileName：coco_display.py\n
@Description：\n
@Author：Wang.Lei\n
@Time：2023/4/3 11:06\n
@Department：Postgrate\n
"""

# from __future__ import print_function
from study.mmdetection_learning.gadl_read import readTiff
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
def show_result(annFile,img_id_num):
    """

    Args:
        annFile: JSON
        img_id_num: 展示image—id（包含在Json中）

    Returns:展示图像

    """
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)      # 设置显示大小

#     annFile=r"H:\daochu_classfied_haved\instances_leaf_train2017.json"
#     annFile = r"H:\daochu_classfied\3channel_tif_voc\train.json"
    coco=COCO(annFile)

# display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))

# imgIds = coco.getImgIds(imgIds = [324158])
    imgIds = coco.getImgIds()
    img = coco.loadImgs(img_id_num)[0]

# 调用图像的路径
    dataDir = "H:\daochu_classfied_haved"
    dataType = 'images'
    I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))[:,:,2:5]
    q = (I-I.min())/(I.max()-I.min())


# load and display instance annotations
# 加载实例掩膜
# catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
# catIds=coco.getCatIds()
    catIds=[]
    for ann in coco.dataset['annotations']:
        if ann['image_id']==imgIds[0]:
            catIds.append(ann['category_id'])

    plt.imshow(q)
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()

annFile = r"H:\daochu_classfied_haved\train.json"
img_id_num = 2
show_result(annFile,img_id_num)

