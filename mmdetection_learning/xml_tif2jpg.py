"""
@FileName：xml_tif2jpg.py\n
@Description：\n
@Author：Wang.Lei\n
@Time：2022/11/23 17:58\n
@Department：Postgrate\n
"""
import xml.etree.ElementTree as ET
import os


xmlpath = r"E:\Desktop\222\Pascal\labels"  #xml标记所在路径
newpath = r"E:\Desktop\222\Pascal\xml-jpg"  # xml保存的新路径
filelist = os.listdir(xmlpath)
# print(filelist)


for xmlfile in filelist:

    xmlnum=int(xmlfile.split('.')[0])
    xmlnum  += 0  # xml的文件的命名
    leng = "0"*(9-len(str(xmlnum)))
    outimg = leng + str(xmlnum) + '.jpg'
    xmlname = leng + str(xmlnum) + '.xml'

    doc = ET.parse(xmlpath + '/'  + xmlfile)
    root = doc.getroot()
    sub1 = root.find('filename')  # 找到filename标签，
    sub2 = root.find('path')   # 找到path标签，
    sub1.text = sub1.text.rstrip('tif')+'jpg'  # 修改filename标签内容
    # sub2.text = pathname # 修改path标签内容
    doc.write(newpath + '/' + xmlname)  # 保存修改




