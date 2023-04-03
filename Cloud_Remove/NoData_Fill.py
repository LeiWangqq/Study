"""
@FileName：NoData_Fill.py\n
@Description：tif图像进行无值填充\n
@Author：Wang.Lei\n
@Time：2023/1/6 15:33\n
@Department：Postgrate\n
"""

import arcpy
from arcpy.sa import *
import os

tif_file_path = r"E:\Desktop\test"  # 图像读取路径
fill_file_path = r"E:\Desktop\create"  # 图像保存路径
os.makedirs(fill_file_path, exist_ok=True)  # 创建路径

arcpy.env.workspace = tif_file_path  # 设置工作路径/根目录

tif_file_name = arcpy.ListRasters("*", "tif")  # 读取所有tif文件

for tif_file in tif_file_name:
    fill_file = arcpy.sa.Con(IsNull(tif_file),
                             # FocalStatistics(tif_file,NbrRectangle(5,5,"CELL"),"MEAN"),   # 设置插值方式
                             FocalStatistics(tif_file, NbrAnnulus(5, 5, "CELL"), "MEAN"),
                             tif_file)
    fill_result_path = fill_file_path + '\\' + tif_file.rstrip(".tif").lstrip(tif_file_path) + "_Fill.tif"  # 保存文件名
    fill_file.save(fill_result_path)  # 保存文件
