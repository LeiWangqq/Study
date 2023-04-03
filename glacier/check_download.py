"""
@FileName：check_download.py\n
@Description：检查下载文件是否完整\n
@Author：Wang.Lei\n
@Time：2023/3/14 10:44\n
@Department：Postgrate\n
"""
from pathlib import Path
import arcpy

def find_files(path, endwith = 'tif'):
    # 获取路径下所有文件中的 .tif文件，并且依据文件名数字进行排序
    # 注意需要 i 需为 Path 类型，才可以使用 .stem 方法访问文件名 str
    find_function = sorted(Path(path).rglob('*' + endwith),
                                key= lambda i: int(i.stem[0:]))
    imgsdir_list = find_function

    return imgsdir_list


arcpy.env.workspace = r"H:\2020-10-01_2020-12-31S2_sorted"
img_path = r"H:\2020-10-01_2020-12-31S2_sorted"
# arcpy.env.workspace = 'H:\Mosaic'
# img_path = r"H:\Mosaic"
# 计算总共渔网个数
shp_path = r"F:\Data\yuwang_data\Southest.shp"
count = int(arcpy.management.GetCount(shp_path).getOutput(0))

# 查找路径下的文件
imgsdir_list = find_files(img_path)
count_list = list(range(count))
# 创建已有文件文件名列表（数字）
imgs_list_num = []
for img_dir in imgsdir_list:
    img_name_num = int(img_dir.stem[0:])
    imgs_list_num.append(img_name_num)

differ_list = list(set(count_list).difference(imgs_list_num))
differ_list.sort(reverse=False)
if __name__=='__main__':
    need_list_ans=[]
    differ_begin = differ_list[0]
    for i in range(len(differ_list)):
        if i+1 < len(differ_list) and differ_list[i+1]-differ_list[i]!=1:
            need_list_ans.append([differ_begin,differ_list[i]])
            print(differ_begin,differ_list[i])
            differ_begin = differ_list[i+1]