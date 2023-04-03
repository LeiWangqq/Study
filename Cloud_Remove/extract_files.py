"""
@FileName：extract_files.py\n
@Description：提取多波段tif图像特定波段组合成新tif\n
@Author：Wang.Lei\n
@Time：2022/12/26 16:55\n
@Department：Postgrate\n
"""
import pathlib
import numpy as np
import tqdm
import imageio
from osgeo import gdal
import warnings

warnings.filterwarnings("ignore")
import shutil


def find_files(path, endwith='tif'):
    '''
    寻找目录文件夹之下所有指定结尾的文件
    Args:
        path: root文件夹路径
        endwith: 结尾文件格式名

    Returns:
    返回【winpath】格式组成的指定文件格式列表
    '''
    initial = []
    find_function = lambda path, endwith: pathlib.Path(path).rglob('*' + endwith)
    path_generator = find_function(path, endwith)
    for i in path_generator:
        initial.append(i)
    ans = initial
    return ans


#   ---------------------------直接复制-----------------------------------------
# data_path =  pathlib.Path(r'H:\2020-06-01_2020-09-30\Synth')
# save_root = r'F:\Synth_compressed'
# pathlib.Path(save_root).mkdir(exist_ok=True,parents=True)
# pbar = tqdm.tqdm(find_files(data_path))
# for i in pbar:
#     if 'compressed' in i.name:
#         save_path = save_root
#         shutil.copy(str(i),save_path +r'\\'+i.parent.name+r'.tif')


#   ----------------------读取对应波段复制-------------------------------
def image_io(img_path, name_path):
    '''
    读取多通道遥感影像，保存真彩色通道为tif影像图片
    Args:
        img_path: 图片路径
        name_path: 图片保存路径及命名

    Returns:
    无【注意】：此方法读取数据会损失空间参考，即转换后tif空间坐标系和原坐标系不同
    '''
    # 保证不对数据做任何变换，读取进出均为 uin16
    # imageio 读取进来为 HWC
    img = imageio.v3.imread(img_path)
    # 若用imageio.v3.imwrite 需要输入 CHW，【注意】此处需要转换
    img_band432 = np.stack((img[:, :, 3], img[:, :, 2], img[:, :, 1]), 2)  # 输出432通道堆叠真彩色
    imageio.imsave(name_path, img_band432)


def readTiff(img_path, name_path):
    dataset = gdal.Open(img_path)  # 输入 img_path 要是 str
    if dataset == None:
        print(img_path + "文件无法打开")
        return
    # -------------------------获取基本行列信息-------------------
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    im_proj = dataset.GetProjection()  # 获取地图投影信息
    im_bands = dataset.RasterCount  # 波段数

    # 获取432波段
    band4 = dataset.GetRasterBand(4)
    band3 = dataset.GetRasterBand(3)
    band2 = dataset.GetRasterBand(2)
    bands = [band4, band3, band2]
    # print('Band Type=', gdal.GetDataTypeName(band1.DataType))  # 输出band的类型
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据，将数据写成数组，对应栅格矩阵,前两个参数是偏移量

    # -------------------------创建tif文件-----------------------------
    driver = gdal.GetDriverByName("GTiff")
    New_dataset = driver.Create(name_path, im_width, im_height, 3, gdal.GDT_Float32)  # 创建新的数据集合,3表示包括三个波段
    New_dataset.SetGeoTransform(im_geotrans)  # 写入仿射矩阵信息
    New_dataset.SetProjection(im_proj)  # 写入地图投影信息
    for i in [1, 2, 3]:  # 写入需要波段信息
        in_band = bands[i - 1]
        in_data = in_band.ReadAsArray()
        out_band = New_dataset.GetRasterBand(i)
        out_band.WriteArray(in_data)
    # 这里注意，创建完成之后，需要删除数据集，不然，后面再次打开的时候，会被占用
    del New_dataset


data_path = pathlib.Path(r'H:\2020-06-01_2020-09-30\Synth')
save_root = r'F:\Synth_compressed'
pathlib.Path(save_root).mkdir(exist_ok=True, parents=True)
pbar = tqdm.tqdm(find_files(data_path))
# for i in pbar:
#     if 'SCENDING' not in i.name:    # 用以判断是否有升降轨Sar影像，去除
#         save_path_name = i.parent.name
#         save_path = save_root+r'\\'+save_path_name
#         pathlib.Path(save_path).mkdir(exist_ok=True,parents=True)
#         readTiff(str(i),save_path +r'\\'+i.name)
#     else:
#         print('done')

for i in pbar:
    if 'compressed' in i.name:
        save_path = save_root
        pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)
        readTiff(str(i), save_path + r'\\' + i.parent.name + r'.tif')
