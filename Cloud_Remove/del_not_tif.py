"""
@FileName：del_not_tif.py\n
@Description：删除指定文件夹中不为tif的文件\n
@Author：Wang.Lei\n
@Time：2023/1/7 19:27\n
@Department：Postgrate\n
"""
import pathlib as Path

root_path = r'H:\2020-06-01_2020-09-30\Synth'

def find_files(path, endwith=''):
    initial = []
    # 寻找当前路径下所有以endwith结尾的文件，此处为空值代表寻找文件夹
    find_function = lambda path, endwith: Path.Path(path).rglob('*' + endwith)

    path_generator = find_function(path, endwith)
    for i in path_generator:
        initial.append(i)
    ans = initial
    return ans
files_list = find_files(root_path)

for i in files_list:
    for j in find_files(i):
        if j.suffix != '.tif':
            j.unlink()
            print(j)

