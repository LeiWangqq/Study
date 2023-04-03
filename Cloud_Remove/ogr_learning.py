"""
@FileName：ogr_learning.py\n
@Description：\n
@Author：Wang.Lei\n
@Time：2022/12/5 15:07\n
@Department：Postgrate\n
"""

"""
from osgeo import ogr
# 读取文件，0是只读，1是可写
data = ogr.Open("China_country_utf8.shp", 0)

layer = data.GetLayer()  # 获取shp文件的图层
dir(layer)  # 图层中可用的方法
layer_number = layer.GetFeatureCount()  # 获取图层中的shp文件个数

feature = layer.GetFeature(0)   # 获取第 0+1 个feature
dir(feature)    # 第一个要素可用的方法
feature.GetField(0)     # 获取第 0+1 个属性名
feat = layer.GetNextFeature()   # 获取下一个要素
layer.ResetReading()    # 复位
"""

import sys
from osgeo import ogr

driver = ogr.GetDriverByName('ESRI Shapefile')          # 设置读取格式
data = driver.Open("E:\Desktop\chinaShp\藏东南.shp", 0)
if data is None:
    print('Could not open')
    sys.exit(1)     # 遇到异常退出程序

layer = data.GetLayer()     # 获取图层
num_Features = layer.GetFeatureCount()      # 获取图层特征数量
print("Features num:", num_Features)
extent = layer.GetExtent()          # 获取图层地理坐标范围
print("Layer Extent\n"
      "Left Down:{0},{1}\n"
      "Right Up:{2},{3}".format(extent[0], extent[2], extent[1], extent[3]))
for i in range(num_Features):
    feature = layer.GetNextFeature()    # 获取特征
    name = feature.GetField(5)          # 获取特征名
    print('name: ',name)
    geometry = feature.GetGeometryRef()     # 获取空间属性定义
    polygon_extent = geometry.GetEnvelope()     # 获取每个空间范围
    print('polygon_extent: ', polygon_extent)
    # feature.Destroy()
layer.ResetReading()

feature_defn = layer.GetLayerDefn()        # 获取图层属性表定义
field = feature_defn.GetFieldCount()        # 获取属性表中字段个数
list_def = feature_defn.GetFieldDefn(0)              # 获取属性表中第1个字段名索引
name = list_def.GetName()

