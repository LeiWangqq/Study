"""
@FileName：gadl_read.py\n
@Description：进行测试图片转换\n
@Author：Wang.Lei\n
@Time：2022/11/15 16:57\n
@Department：Postgrate\n
"""

import pathlib
import numpy as np
from PIL import Image
from osgeo import gdal
import cv2
import tqdm
import matplotlib.pyplot as plt
import imageio



# --------------------------遍历文件夹下所有文件-------------------------------

def find_files(path, endwith='tif'):
    initial = []
    find_function = lambda path, endwith: pathlib.Path(path).rglob('*' + endwith)
    path_generator = find_function(path, endwith)
    for i in path_generator:
        initial.append(i)
    ans = initial
    return ans


def remov_files(path, endwith):
    files = find_files(path, '.' + endwith)
    for i in tqdm.trange(len(files)):
        files[i].unlink()


def readTiff(img_path):
    dataset = gdal.Open(img_path)  # 输入 img_path 要是 str
    if dataset == None:
        print(img_path + "文件无法打开")
        return
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset.RasterCount  # 波段数   高分影像四波段【RGBA】

    band1 = dataset.GetRasterBand(1)
    # print(im_bands)
    # print('Band Type=', gdal.GetDataTypeName(band1.DataType))  # 输出band的类型
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据，将数据写成数组，对应栅格矩阵,前两个参数是偏移量
    im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    im_proj = dataset.GetProjection()  # 获取地图投影信息
    # print(im_data.shape)
    return im_data, im_geotrans, im_proj


def tif_pil_store(img_path, name_path, endwith):
    img_path = str(img_path)  # pathlib 路径转换字符串
    dataset = gdal.Open(img_path)  # gdal打开图像获取信息
    band = []
    for i in range(dataset.RasterCount):
        band.append(dataset.GetRasterBand(i + 1).ReadAsArray())
    BAND = np.array(band)
    BAND = BAND.transpose(1, 2, 0)  # 将图像从 CHW 转换为 HWC

    # 转换图片
    BAND = (BAND - BAND.min()) / (BAND.max() - BAND.min())  # 将图像转换为 0-1 的 float
    band = np.uint8(BAND * 255)  # 转换为 uint8
    im = Image.fromarray(band[:, :, 0:3])  # 输出前三通道
    im.save(name_path+endwith)


def cv_store(img_path, name_path, endwith):
    # cv保存 默认读取保存深度均为 uint8，即图像有损失值
    img_path = str(img_path)  # pathlib 路径转换字符串
    img = cv2.imread(img_path)
    cv2.imwrite(name_path+endwith,img)
    # 【注意】读取进来为三通道，且为BGR，cv2.imwrite也是BGR形式


def image_io(img_path, name_path, endwith):
    # 保证不对数据做任何变换，读取进出均为 uin16
    # imageio 读取进来为 HWC
    img = imageio.v3.imread(img_path)
    # 若用imageio.v3.imwrite 需要输入 CHW，【注意】此处需要转换
    # img_band123 = img[:,:,0:3]    # 输出前三通道
    img_band123 = img[:, :, 1:4]
    # img_band123 = img[:,:,1]
    imageio.imsave(name_path+endwith,img_band123)



def gdalstore(img_path, name_path, endwith):
    dataset = gdal.Open(img_path)

    im_width = dataset.RasterXSize           # 栅格矩阵的列数
    im_height = dataset.RasterYSize          #栅格矩阵的行数
    m_bands = dataset.RasterCount            #波段数

    im_geotrans = dataset.GetGeoTransform()  #获取仿射矩阵信息
    im_proj = dataset.GetProjection()        #获取投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据，读为一个array，[C，H，W]，0代表坐标起点

    # 取出B、G、R波段存入band_list，不能直接切片如:im_data[1,:,:],会丢失许多信息
    band_list = []
    for i in range(m_bands - 1):
        # GeoTiff波段从1开始计数
        band= dataset.GetRasterBand(i + 1)
        band_list.append(band)

    driver = gdal.GetDriverByName("GTiff")  #  保存为.tif格式的GeoTIFF文件
    # 新的栅格图层格式：路径、行列数、波段数与数据格式
    datasetnew = driver.Create(name_path+ endwith ,
                               im_width,
                               im_height,
                               m_bands - 1,
                               # 1,
                               gdal.GDT_UInt16,)

    datasetnew.SetGeoTransform(im_geotrans)  # 写入仿射矩阵信息
    datasetnew.SetProjection(im_proj)   # 写入投影信息
    datasetnew.WriteRaster(0,0,im_width,im_height,im_data.tobytes(),im_width,im_height,band_list = [3,2,1]) # 读入像素信息，波段排列为rgb
    datasetnew.FlushCache()     # 写入磁盘
    datasetnew = None         # 其从内存空间中释放，完成写入与保存工作。


# ------------------------*调用*------------------------------
if __name__ == '__main__':
    path = r'H:\daochu_classfied\images'
    path_name = path.rstrip('images')
    img_files = find_files(path, 'tif')
    save_path = path_name  + r'3channel_tif'
    endwith = '.tif'
    # img_0 = str(img_files[0])
    # dataset0 = gdal.Open(img_0)
    for i in tqdm.trange(len(img_files)):
        img = img_files[i]
        pathlib.Path(path_name + endwith).mkdir(parents=False,exist_ok=True) # exist_ok 决定目录存在时是否报错
        path_locate = save_path  + endwith
        name = path_locate + '\\' + img.stem
        #  获取正确路径方便调用
        # tif_pil_store(img, name, '.' + endwith)    # PIL保存 jpg
        # cv_store(img, name, '.' + endwith)        # cv2 保存 png
        image_io(img, name, '.' + endwith)         # imageio 保存 tif
        # gdalstore(img, name, '.' + endwith)



# 使用CV2，显示tif图像
def showTiff(img_path):
    img = cv2.imread(img_path, 4)
    # 第二个参数是通道数和位深的参数，
    # IMREAD_UNCHANGED = -1  # 不进行转化，比如保存为了16位的图片，读取出来仍然为16位。
    # IMREAD_GRAYSCALE = 0  # 进行转化为灰度图，比如保存为了16位的图片，读取出来为8位，类型为CV_8UC1。
    # IMREAD_COLOR = 1  # 进行转化为RGB三通道图像，图像深度转为8位
    # IMREAD_ANYDEPTH = 2  # 保持图像深度不变，进行转化为灰度图。
    # IMREAD_ANYCOLOR = 4  # 若图像通道数小于等于3，则保持原通道数不变；若通道数大于3则只取取前三个通道。图像深度转为8位
    # print(img)  # 输出图像的数据
    # print(img.shape)  # 输出图像的三维
    # print(img.dtype)  # 输出图像的编码类型
    # print(img.min())  # 输出图像的最小值
    # print(img.max())  # 输出图像的最大值
    # 创建窗口并显示图像
    cv2.namedWindow("Image")  # 创建一个窗口
    cv2.imshow("Image", img)  # 显示图像
    cv2.waitKey(0)  # 设置显示图像的延迟
    # 释放窗口
    cv2.destroyAllWindows()  # 【特别注意】 此处推出用 ESC 不要直接关闭 window 否则程序会暂停
