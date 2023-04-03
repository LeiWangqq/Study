"""
@FileName：rename.py\n
@Description：将获取到的切片栅格重命名\n
@Author：Wang.Lei\n
@Time：2023/3/14 9:49\n
@Department：Postgrate\n
"""
from pathlib import Path
import shutil

def find_files(path, endwith = 'tif'):
    # 获取路径下所有文件中的 .tif文件，并且依据文件名数字进行排序
    # 注意需要 i 需为 Path 类型，才可以使用 .stem 方法访问文件名 str
    find_function = sorted(Path(path).rglob('*' + endwith),
                                key= lambda i: int(i.stem[0:]))
    imgsdir_list = find_function

    return imgsdir_list

# imgs_dir = r'H:\2020-10-01_2020-12-31S2_sorted'
imgs_dir = r'H:\导出_已分类'


img_dir_list  = find_files(imgs_dir)

if __name__ == '__main__':
    for i in ['images','labels']:
        save_folder = Path(f'H:\daochu_classfied\{i}')
        Path(save_folder).mkdir(parents=True,exist_ok=True)
        img_list = find_files(Path(imgs_dir)/f'{i}')
        for j in img_list:
            shutil.copy(j,str(save_folder)+f'\{int(j.stem)}.tif')


''' 迁移数据使用
    # length = len(img_dir_list)
    # root = Path('H:\mosaic')
    # start_num = None
    # for i in range(length):
    #     img_name_num = int(img_dir_list[i].stem)
    #     print(img_dir_list[i].stem)
    #     if img_name_num%1000 == 0:
    #         print(img_name_num)
    #         start_num = img_name_num
    #     else:
    #         start_num = img_name_num-img_name_num%1000
    #     end_num = start_num+999
    #     save_folder = Path(root)/f'{start_num}_to_{end_num}'
    #     Path(save_folder).mkdir(parents=True,exist_ok=True)
    #     if start_num == None:
    #         continue
    #     if img_name_num in range(start_num,end_num+1):
    #         print(img_dir_list[i],save_folder / img_dir_list[i].name)
    #         shutil.copy(img_dir_list[i],save_folder / img_dir_list[i].name)
'''