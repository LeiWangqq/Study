"""
@FileName：merge_tif.py\n
@Description：合并切片的tif图像\n
@Author：Wang.Lei\n
@Time：2022/12/27 14:37\n
@Department：Postgrate\n
"""
from osgeo import gdal
import numpy as np
import pathlib

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

# -*- coding: utf-8 -*-
"""
 @ time: 2022/11/11 13:49
 @ file:
 @ author: QYD2001
"""
import pathlib

import numpy as np
from osgeo import gdal




