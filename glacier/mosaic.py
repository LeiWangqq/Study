"""
@FileName：mosaic.py\n
@Description：\n
@Author：Wang.Lei\n
@Time：2023/3/13 18:59\n
@Department：Postgrate\n
"""




# In[arcpy镶嵌]

# import pathlib
# import arcpy
# arcpy.env.workspace = r"J:\2020-10-01_2020-12-31S2\test-2"
# rasters = arcpy.ListRasters("*", "tif")
#
# mosaic_rasters=""
# for raster in rasters:
#     mosaic_rasters=mosaic_rasters+raster+";"
# print(mosaic_rasters)
#
# arcpy.management.MosaicToNewRaster(mosaic_rasters,
#                                    r'J:\save',
#                                    r'test.tif',
#                                    "",
#                                    r'64_BIT', "", 11,
#                                    r'MEAN', r'MATCH')

# In[gdal 镶嵌]
# !/usr/bin/env python
# coding: utf-8

from osgeo import gdal
import os
import glob

import numpy as np
import math

# In[gdal]
from osgeo import gdal

def read_img(filename):
    '''
    读取影像为数组并返回信息
    ——————————————————————————
    @param
        filename ：输入的影像路径
    @return:
        影像的numpy数组格式，并显示影像的基本信息
    '''
    dataset = gdal.Open(filename)  # 打开文件
    print('栅格矩阵的列数:', dataset.RasterXSize)
    print('栅格矩阵的行数:', dataset.RasterYSize)
    print('波段数:', dataset.RasterCount)
    print('数据类型:', dataset.GetRasterBand(1).DataType)
    print('仿射矩阵(左上角像素的大地坐标和像素分辨率)', dataset.GetGeoTransform())
    print('地图投影信息:', dataset.GetProjection())
    im_data = dataset.ReadAsArray()
    del dataset

    return im_data


def read2arr(filename):  # 读取影像为数组
    '''
    读取影像为数组
    ——————————————————————————
    @param
        filenam ：输入的影像路径
    @return:
        影像的numpy数组格式
    '''
    dataset = gdal.Open(filename)  # 打开文件
    im_data = dataset.ReadAsArray()
    del dataset
    return im_data


def ds2tif(dataset, out_fn):
    """
    将GDAL dataset数据格式写入tif保存
    ——————————————————————————
    @param：
        dataset：输入的GDAL影像数据格式
        out_fn：输出的文件路径
    @return：
        输出影像文件

    """
    # 读取dataset信息
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_datatype = dataset.GetRasterBand(1).DataType

    # 将dataset 写入 tif
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(out_fn, im_width, im_height, im_bands, im_datatype)
    ds.SetGeoTransform(im_geotrans)
    ds.SetProjection(im_proj)
    ds.GetRasterBand(1).WriteArray(dataset.ReadAsArray())

    del ds


def arr2tif(arr, out_fn, Transform=None, Projection=None, Band=1, Datatype=6):
    """
    将数组格式写入tif保存
    ——————————————————————————
    @param：
        arr：待保存的影像数组
        out_fn：输出的文件路径
        Transform：仿射矩阵六参数数组，默认为空,详细格式见GDAL。
        Projection ：投影，默认为空,详细格式见GDA
        Band ：波段数，默认为1
        Datatype：保存数据格式（位深），默认为6，GDT_Float32
    @return：
        输出影像文件

    """
    (x, y) = arr.shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(out_fn, y, x, Band, Datatype)
    if not Transform == None:
        ds.SetGeoTransform(Transform)
    if not Projection == None:
        ds.SetProjection(Projection)
    ds.GetRasterBand(Band).WriteArray(arr)
    del ds


def GetExtent(in_fn):
    '''
    计算影像角点的地理坐标或投影坐标
    ——————————————————————————
    @param：
        影像路径
    @return:
        min_x： x方向最小值
        max_y： y方向最大值
        max_x： x方向最大值
        min_y:  y方向最小值
    '''
    ds = gdal.Open(in_fn)
    geotrans = list(ds.GetGeoTransform())
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    min_x = geotrans[0]
    max_y = geotrans[3]
    max_x = geotrans[0] + xsize * geotrans[1]
    min_y = geotrans[3] + ysize * geotrans[5]
    ds = None

    return min_x, max_y, max_x, min_y


def DsGetExtent(ds):
    '''
    读取dataset格式，计算影像角点的地理坐标或投影坐标
    ——————————————————————————
    @param：
        ds： GDAL dataset格式数据
    @return:
        min_x： x方向最小值
        max_y： y方向最大值
        max_x： x方向最大值
        min_y:  y方向最小值
    '''
    geotrans = list(ds.GetGeoTransform())
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    min_x = geotrans[0]
    max_y = geotrans[3]
    max_x = geotrans[0] + xsize * geotrans[1]
    min_y = geotrans[3] + ysize * geotrans[5]
    ds = None

    return min_x, max_y, max_x, min_y


def pix2geo(Xpixel, Ypixel, GeoTransform):
    '''
    计算影像某一像素点的地理坐标或投影坐标
    ——————————————————————————
    @param：
        Xpixel ：像素坐标x
        Ypixel： 像素坐标y
        GeoTransform：仿射变换参数
    @return:
        XGeo： 地理坐标或投影坐标X
        YGeo： 地理坐标或投影坐标Y
    '''
    XGeo = GeoTransform[0] + GeoTransform[1] * Xpixel + Ypixel * GeoTransform[2]
    YGeo = GeoTransform[3] + GeoTransform[4] * Xpixel + Ypixel * GeoTransform[5]
    return XGeo, YGeo


def geo2pix(dataset, x, y):
    '''
    根据GDAL的仿射变换参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    ————————————————————————————————
    @param
        dataset: GDAL地理数据
        x: 投影或地理坐标x
        y: 投影或地理坐标y
    @return:
        影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])

    return np.linalg.solve(a, b)


def Mosaic_all(path):
    '''
    将指定路径文件夹下的tif影像全部镶嵌到一张影像上
    ————————————————————————————————
    @param
        path：tif影像存放路径
    @return:
        镶嵌合成的整体影像
    '''
    os.chdir(path)
    # 如果存在同名影像则先删除
    if os.path.exists('mosaiced_image.tif'):
        os.remove('mosaiced_image.tif')
    in_files = glob.glob("*.TIF")
    in_fn = in_files[0]
    # 获取待镶嵌栅格的最大最小的坐标值
    min_x, max_y, max_x, min_y = GetExtent(in_fn)
    for in_fn in in_files[1:]:
        minx, maxy, maxx, miny = GetExtent(in_fn)
        min_x = min(min_x, minx)
        min_y = min(min_y, miny)
        max_x = max(max_x, maxx)
        max_y = max(max_y, maxy)

    # 计算镶嵌后影像的行列号
    in_ds = gdal.Open(in_files[0])
    geotrans = list(in_ds.GetGeoTransform())
    width = geotrans[1]
    height = geotrans[5]

    columns = math.ceil((max_x - min_x) / width)
    rows = math.ceil((max_y - min_y) / (-height))
    in_band = in_ds.GetRasterBand(1)

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create('mosaiced_image.tif', columns, rows, 1, in_band.DataType)
    out_ds.SetProjection(in_ds.GetProjection())
    geotrans[0] = min_x
    geotrans[3] = max_y
    out_ds.SetGeoTransform(geotrans)
    out_band = out_ds.GetRasterBand(1)

    # 定义仿射逆变换
    inv_geotrans = gdal.InvGeoTransform(geotrans)

    # 开始逐渐写入
    for in_fn in in_files:
        in_ds = gdal.Open(in_fn)
        in_gt = in_ds.GetGeoTransform()
        # 仿射逆变换
        offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])
        x, y = map(int, offset)
        print(x, y)
        trans = gdal.Transformer(in_ds, out_ds, [])  # in_ds是源栅格，out_ds是目标栅格
        success, xyz = trans.TransformPoint(False, 0, 0)  # 计算in_ds中左上角像元对应out_ds中的行列号
        x, y, z = map(int, xyz)
        print(x, y, z)
        data = in_ds.GetRasterBand(1).ReadAsArray()
        out_band.WriteArray(data, x, y)  # x，y是开始写入时左上角像元行列号
    del in_ds, out_band, out_ds
    return 0


def Mosaic(ds1, ds2, path):
    '''
    将两幅影像镶嵌至同一幅影像
    ————————————————————————————————
    @param
        ds1：镶嵌数据集1
        ds2：镶嵌数据集1
    @return:
        镶嵌合成的整体影像
    '''
    band1 = ds1.GetRasterBand(1)
    rows1 = ds1.RasterYSize
    cols1 = ds1.RasterXSize

    band2 = ds2.GetRasterBand(1)
    rows2 = ds2.RasterYSize
    cols2 = ds2.RasterXSize

    (minX1, maxY1, maxX1, minY1) = DsGetExtent(ds1)
    (minX2, maxY2, maxX2, minY2) = DsGetExtent(ds2)

    transform1 = ds1.GetGeoTransform()
    pixelWidth1 = transform1[1]
    pixelHeight1 = transform1[5]  # 是负值（important）

    transform2 = ds2.GetGeoTransform()
    pixelWidth2 = transform1[1]
    pixelHeight2 = transform1[5]

    # 获取输出图像坐标
    minX = min(minX1, minX2)
    maxX = max(maxX1, maxX2)
    minY = min(minY1, minY2)
    maxY = max(maxY1, maxY2)

    # 获取输出图像的行与列
    cols = int((maxX - minX) / pixelWidth1)
    rows = int((maxY - minY) / abs(pixelHeight1))

    # 计算图1左上角的偏移值（在输出图像中）
    xOffset1 = int((minX1 - minX) / pixelWidth1)
    yOffset1 = int((maxY1 - maxY) / pixelHeight1)

    # 计算图2左上角的偏移值（在输出图像中）
    xOffset2 = int((minX2 - minX) / pixelWidth1)
    yOffset2 = int((maxY2 - maxY) / pixelHeight1)

    # 创建一个输出图像
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(path, cols, rows, 1, band1.DataType)  # 1是bands，默认
    out_band = out_ds.GetRasterBand(1)

    # 读图1的数据并将其写到输出图像中
    data1 = band1.ReadAsArray(0, 0, cols1, rows1)
    out_band.WriteArray(data1, xOffset1, yOffset1)

    # 读图2的数据并将其写到输出图像中
    data2 = band2.ReadAsArray(0, 0, cols2, rows2)
    out_band.WriteArray(data2, xOffset2, yOffset2)
    ''' 写图像步骤'''

    # 第二个参数是1的话：整幅图像重度，不需要统计
    # 设置输出图像的几何信息和投影信息
    geotransform = [minX, pixelWidth1, 0, maxY, 0, pixelHeight1]
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(ds1.GetProjection())

    del ds1, ds2, out_band, out_ds, driver

    return 0


def raster_overlap(ds1, ds2, nodata1=None, nodata2=None):
    '''
    两个栅格数据集取重叠区或求交集  （仅测试方形影像）
    ————————————————————————————————
    @param:
        ds1 (GDAL dataset) - GDAL dataset of an image
        ds2 (GDAL dataset) - GDAL dataset of an image
        nodata1 (number) - nodata value of image 1
        nodata2 (number) - nodata value of image 2
    @returns:
        ds1c (GDAL dataset), ds2c (GDAL dataset): intersection dataset1 and intersection dataset2.
    '''
    ##Setting nodata
    nodata = 0
    ###Check if images NoData is set
    if nodata2 is not None:
        nodata = nodata2
        ds2.GetRasterBand(1).SetNoDataValue(nodata)
    else:
        if ds2.GetRasterBand(1).GetNoDataValue() is None:
            ds2.GetRasterBand(1).SetNoDataValue(nodata)

    if nodata1 is not None:
        nodata = nodata1
        ds1.GetRasterBand(1).SetNoDataValue(nodata1)
    else:
        if ds1.GetRasterBand(1).GetNoDataValue() is None:
            ds1.GetRasterBand(1).SetNoDataValue(nodata)

    ### Get extent from ds1
    projection = ds1.GetProjection()
    geoTransform = ds1.GetGeoTransform()

    ###Get minx and max y

    [minx, maxy, maxx, miny] = DsGetExtent(ds1)
    [minx_2, maxy_2, maxx_2, miny_2] = DsGetExtent(ds2)

    min_x = sorted([maxx, minx_2, minx, maxx_2])[1]  # 对边界值排序，第二三个为重叠区边界
    max_y = sorted([maxy, miny_2, miny, maxy_2])[2]
    max_x = sorted([maxx, minx_2, minx, maxx_2])[2]
    min_y = sorted([maxy, miny_2, miny, maxy_2])[1]

    ###Warp to same spatial resolution
    gdaloptions = {'format': 'MEM', 'xRes': geoTransform[1], 'yRes':
        geoTransform[5], 'dstSRS': projection}
    ds2w = gdal.Warp('', ds2, **gdaloptions)
    ds2 = None

    ###Translate to same projection
    ds2c = gdal.Translate('', ds2w, format='MEM', projWin=[min_x, max_y, max_x, min_y],
                          outputSRS=projection)
    ds2w = None
    ds1c = gdal.Translate('', ds1, format='MEM', projWin=[min_x, max_y, max_x, min_y],
                          outputSRS=projection)
    ds1 = None

    return ds1c, ds2c

from osgeo import gdal,gdalconst
from pathlib import Path
def find_files(path, endwith = 'tif'):
    # 获取路径下所有文件中的 .tif文件，并且依据文件名数字进行排序
    # 注意需要 i 需为 Path 类型，才可以使用 .stem 方法访问文件名 str
    find_function = sorted(Path(path).rglob('*' + endwith), key= lambda i: int(i.stem))
    imgsdir_list = []
    for i in find_function:
        imgsdir_list.append(str(i))
    return imgsdir_list

def RasterMosaic(inputfilePath,img_list,name):
    print("--------begin---------")
    input_img1 = gdal.Open(inputfilePath, gdal.GA_ReadOnly) # 第一幅影像
    inputProj1 = input_img1.GetProjection()
    outputfilePath = f'H:/save/{name}.tif'
    options=gdal.WarpOptions(srcSRS=inputProj1, dstSRS=inputProj1,format='GTiff',resampleAlg=gdalconst.GRA_Bilinear)
    gdal.Warp(outputfilePath,img_list,options=options)
    print("--------end---------")

tif_path = r'H:\2020-10-01_2020-12-31S2_sorted'
tif_files = find_files(tif_path)
RasterMosaic(tif_files[0],tif_files[0:1000],'0_1000')
