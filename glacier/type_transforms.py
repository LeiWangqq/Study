"""
@FileName：type_transforms.py\n
@Description：转换tif文件数据类型，并进行arcpy镶嵌图像\n
@Author：Wang.Lei\n
@Time：2023/3/17 17:28\n
@Department：Postgrate\n
"""
from osgeo import gdal
import os
import numpy as np
from pathlib import Path
# In[重构图像格式]
def redtif(path):
    '''
    :param path: tif图像路径
    :return:
    '''
    dataset = gdal.Open(path)
    tif_xsize = dataset.RasterXSize
    tif_ysize = dataset.RasterYSize
    tif_geo_transform = dataset.GetGeoTransform()
    tif_prj = dataset.GetProjection()
    # ReadAsArray()
    tif_data = dataset.ReadAsArray(0, 0, tif_xsize, tif_ysize)
    return tif_data, tif_prj, tif_geo_transform

# In[镶嵌16位深]
class tiffff:
    def __init__(self,path):
        self.path = path
        self.tif_data, self.tif_prj, self.tif_geo_transform = redtif(self.path)
        # 转换类型为int8
        self.tif_data = np.int8((self.tif_data-self.tif_data.min())/(self.tif_data.max()-self.tif_data.min())*255)

    def savetif(self,savepath,name):
        driver = gdal.GetDriverByName('GTiff')
        save_dir = os.path.join(savepath,name)
        tif_band_size,tif_new_xsize,tif_new_ysize =  self.tif_data.shape[0], self.tif_data.shape[1],self.tif_data.shape[2]
        # 转换格式为float64
        # dataset_new = driver.Create(save_dir , tif_new_ysize,tif_new_xsize,tif_band_size,gdal.GDT_Float64)
        dataset_new = driver.Create(save_dir, tif_new_ysize, tif_new_xsize, tif_band_size, gdal.GDT_Byte)

        dataset_new.SetGeoTransform(self.tif_geo_transform)
        dataset_new.SetProjection(self.tif_prj)
        for i in range(tif_band_size):
            dataset_new.GetRasterBand(i+1).WriteArray(self.tif_data[i])
        del dataset_new


# path = r"H:\1\5.tif"
# savepath = r"H:\save"
# name = 'test2221.tif'
# tiffff(path).savetif(savepath,name)

# In[arcpy镶嵌位深]
##==================================
##Mosaic To New Raster
##Usage: MosaicToNewRaster_management inputs;inputs... output_location raster_dataset_name_with_extension
##                                    {} 8_BIT_UNSIGNED | 1_BIT | 2_BIT | 4_BIT
##                                    | 8_BIT_SIGNED | 16_BIT_UNSIGNED | 16_BIT_SIGNED | 32_BIT_FLOAT | 32_BIT_UNSIGNED
##                                    | 32_BIT_SIGNED | | 64_BIT {cellsize} number_of_bands {LAST | FIRST | BLEND  | MEAN
##                                    | MINIMUM | MAXIMUM} {FIRST | REJECT | LAST | MATCH}



import arcpy
root_path = r'H:\Mosaic_Tif'        # 需镶嵌文件夹
save_path = r'H:\Mosaic_Tif'    # 镶嵌后保存文件夹
# root_path = r'H:\Mosaic_Tif\thousand_order'        # 需镶嵌文件夹
# save_path = r'H:\Mosaic_Tif'    # 镶嵌后保存文件夹
Path(save_path).mkdir(parents=True,exist_ok=True)

# 按根文件夹中子文件夹名称进行排序
# # 【注意】镶嵌时为了不出现无值，类型选择MAXIMUM
# for i in sorted(Path(root_path).glob("*"),key= lambda i:int(i.name.split('_to_')[0])):
#     # if int(i.name.split('_to_')[1]):
#     if int(i.name.split('_to_')[1]) > 11000:
#         arcpy.env.workspace = str(i)
#     # 设置工作空间并从中获取后缀为 .tif的所有文件
#         name= arcpy.ListFiles("*.tif")
#         print(len(name),i.name)
#         if name != []:
#         ##Mosaic several TIFF images to a new TIFF image
#             arcpy.MosaicToNewRaster_management(name,str(save_path), f"{i.name}.tif", "",\
#                                         "16_BIT_UNSIGNED", "", "11", "MAXIMUM","FIRST")
        

# 按文件夹名称进行排序
# 【注意】镶嵌时为了不出现无值，类型选择max

arcpy.env.workspace = str(root_path)         # 设置工作空间并从中获取后缀为 .tif的所有文件
name= arcpy.ListFiles("*.tif")
# if name != []:
#     ##Mosaic several TIFF images to a new TIFF image
#     arcpy.MosaicToNewRaster_management(name,str(save_path), f"all.tif", "",\
#                                         "16_BIT_UNSIGNED", "", "11", "MAXIMUM","FIRST")