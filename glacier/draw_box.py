"""
@FileName：draw_box.py\n
@Description：画出描述框\n
@Author：Wang.Lei\n
@Time：2023/3/31 16:49\n
@Department：Postgrate\n
"""
import cv2
from osgeo import gdal
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from skimage import io
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

def readTiff(img_path):
    dataset = gdal.Open(img_path)  # 输入 img_path 要是 str
    if dataset == None:
        print(img_path + "文件无法打开")
        return
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据，将数据写成数组，对应栅格矩阵,前两个参数是偏移量
    return im_data.transpose(1,2,0)  # 维度设置为H-W-C，cv显示


def draw_bbox(img: Image.Image,
              bbox: Tuple[float, float, float, float],
              prob: float,
              rect_color: Tuple[int, int, int] = (255, 0, 0),
              text: Optional[str] = None,
              better_font: Optional[str] = None):
    img_draw = ImageDraw.Draw(img, 'RGBA')
    x1, y1, x2, y2 = bbox
    if better_font is not None:
        font = ImageFont.truetype(
            better_font,
            12,
        )
    else:
        font = ImageFont.load_default()

    img_draw.rectangle((x1 - 2, y1 - 2, x2 + 2, y2 + 2),
                       outline=rect_color,
                       width=2)

    # Show class label on the top right corner
    if text is not None:
        tw, th = font.getsize(text)
        img_draw.rectangle((x2 - tw, y1, x2, y1 + th), fill='black')
        img_draw.text((x2 - tw, y1), text, font=font, anchor='rt')

    # Show probablity of top left corner
    tw, th = font.getsize(f'{prob:.2f}')
    img_draw.rectangle((x1, y1, x1 + tw, y1 + th), fill='black')
    img_draw.text((x1, y1), f'{prob:.2f}', font=font)




def draw_box(id_image, json_path, img_path):
    coco = COCO(json_path)

    # list_imgIds = coco.getImgIds()    # 获取含有该给定类别的所有图片的id
    img = coco.loadImgs(id_image)[0]   # 获取满足上述要求，并给定显示第num幅image对应的dict
    # image = io.imread(img_path + img['file_name'])  # 读取图像

    image_name = img['file_name']  # 读取图像名字
    image = readTiff(img_path + r'\\' + image_name).copy()

    image_id = img['id']  # 读取图像id
    img_annIds = coco.getAnnIds(imgIds=[image_id])  # 获取图像id对应的ann——id
    img_anns = coco.loadAnns(ids=img_annIds)    # 获取的ann——id加载ann 列表数组

    for i in range(len(img_anns)):
        x, y, w, h = img_anns[i]['bbox']  # 读取边框
        image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
        image = cv2.putText(image, '0.8', (int(x), int(y)),cv2.FONT_ITALIC, 11/w,(97, 0, 255))

    # plt.rcParams['figure.figsize'] = (6, 6)
    # # 此处的20.0是由于我的图片是2000*2000，目前还没去研究怎么利用plt自动分辨率。
    plt.imshow(image)
    plt.show()




if __name__ == '__main__':
    id_image = 5746  # 要对应json文件中包含的image——id
    json_path =r'H:\daochu_classfied\3channel_tif_voc\test.json'
    img_path = r'H:\daochu_classfied\3channel_tif_voc\JPEGImages'
    draw_box(id_image,r'H:\daochu_classfied\3channel_tif_voc\val.json',r'H:\daochu_classfied\3channel_tif_voc\JPEGImages')

