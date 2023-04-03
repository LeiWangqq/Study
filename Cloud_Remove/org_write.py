"""
@FileName：org_write.py.py\n
@Description：读取栅格字段属性表\n
@Author：Wang.Lei\n
@Time：2022/12/5 21:30\n
@Department：Postgrate\n
"""
from osgeo import ogr
import sys

data = "E:\Desktop\chinaShp\藏东南.shp"
driver = ogr.GetDriverByName('ESRI Shapefile')
data = driver.Open(data, 0)
layer = data.GetLayer()
attr_list_def = layer.GetLayerDefn()
attr_list_num = attr_list_def.GetFieldCount()

name_all = []

def str_length(str,length=30):
    """
    输入一个str，返回一定长度的str，便于write
    Args:
        str: 输入字符
        length：返回长度
    Returns:

    """
    len_input = len(str)
    if len_input < length:
        num = length - len_input
        str = ' '*num + str
    return str

tplt = "{:<40}\t"  # 定义格式
txt_sa = '%-20s\t'

out_txt = open('藏东南.txt', 'w+', encoding= 'utf-8')
for i in range(attr_list_num):
    name_defn = attr_list_def.GetFieldDefn(i)
    name = name_defn.GetName()
    name_all.append(name)
    out_txt.write(txt_sa % name)
out_txt.write('\n')


feature = layer.GetNextFeature()
while feature:
    for i in range(attr_list_num):
        name_all[i] = feature.GetField(i)
        out_txt.write(txt_sa % str(name_all[i]))
    out_txt.write('\n')
    feature = layer.GetNextFeature()

print('done')
out_txt.close()         # 不关闭将不会写入文件