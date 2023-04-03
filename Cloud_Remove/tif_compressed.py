"""
@FileName：tif_compressed.py\n
@Description：对文件夹中每个tif文件都进行压缩，减少tif图像大小\n
@Author：Wang.Lei\n
@Time：2023/1/6 16:22\n
@Department：Postgrate\n
"""
import rasterio as rio
import rasterio
import os
from tqdm import tqdm
import pandas as pd

# 搜寻图像
DF = pd.read_csv(r"H:\2020-06-01_2020-09-30.csv")
Positive = DF[DF['pos_neg'] == 1]
Synth_positive = Positive[Positive['part'] == 'Synth']
Synth_positive = Synth_positive.reset_index(drop=False)
Parent_path = r'H:\2020-06-01_2020-09-30\Synth'

for i in range(len(Synth_positive)):
    print(Synth_positive.loc[i].num)
    Negtive = DF[(DF['pos_neg'] == 0) & (DF['num'] == Synth_positive.loc[i].num)].iloc[0]
    input_tif = os.path.join(Parent_path, f'{Synth_positive.loc[i].num:05d}',
                              Synth_positive.loc[i].tif).rstrip('Cloud.tif') + 'Reprj_fillnodata_cloud.tif'
    Output_tif = os.path.join(Parent_path, f'{Synth_positive.loc[i].num:05d}', 'Reprj_cloud_compressed.tif')

    rasterfile = input_tif
    # 打开栅格
    rasterdata = rio.open(rasterfile)
    # 读取栅格
    rasterread = rasterdata.read()
    # 获取栅格信息
    profile = rasterdata.profile
    print(profile)
    # 选择压缩方式
    profile.update(
        compress='DEFLATE',  # 压缩方式：这里设置和原cloud图相同
    )
    # 导出文件路径与名字
    out_put_name = Output_tif
    # 导出
    with rasterio.open(out_put_name, mode='w', **profile) as dst:
        dst.write(rasterread)









    # #   设置输入输出文件夹
# Input_path = r"E:\Desktop\create" + "\\"
# Output_path = r"E:\Desktop\compressed" + "\\"
# os.makedirs(Output_path, exist_ok= True)

# #   读取tif文件列表
# pathDir = os.listdir(Input_path)
# tif_dir = []
# for dataname in pathDir:
#     if os.path.splitext(dataname)[1] == '.tif':
#         tif_dir.append(dataname)

# # 压缩函数
# for i in tqdm(range(len(tif_dir))):
#     # 读入栅格文件
#     rasterfile = Input_path + "\\" + tif_dir[i]
#     # 打开栅格
#     rasterdata = rio.open(rasterfile)
#     # 读取栅格
#     rasterread = rasterdata.read()
#     # 获取栅格信息
#     profile = rasterdata.profile
#     print(profile)
#     # 选择压缩方式
#     profile.update(
#         compress='DEFLATE',  # 压缩方式：这里设置和原cloud图相同
#     )
#     # 导出文件路径与名字
#     out_put_name = Output_path + pathDir[i].rstrip('.tif') + 'compressed.tif'
#     # 导出
#     with rasterio.open(out_put_name, mode='w', **profile) as dst:
#         dst.write(rasterread)
