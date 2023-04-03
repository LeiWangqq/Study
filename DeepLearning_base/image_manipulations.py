#   方法一：用matplotlib.pyplot 读取
"""
import matplotlib.pyplot as plt
import matplotlib.image as img

Image_load = img.imread('E:\Pictures\lock_picture.png')
# 存储格式为 [R,G,B,A] 四层
# 读取的数据类型为归一化后数据即 [R,G,B,A]/255
Image_shape = Image_load.shape
# 输出个数为 [weight, height, channel]

plt.imshow(Image_load)
plt.show()
"""

# 方法二：PIL获取图像的数值矩阵
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

Image_load = Image.open('E:\Pictures\lock_picture.png')
print(type(Image_load)) # 返回Image对象
# <class 'PIL.PngImagePlugin.JpegImageFile'>

# 对象的属性来查看图片的信息
print(Image_load.format) # 返回图像的格式  # JPEG

print(Image_load.size) # 返回图像的尺寸    # (1920, 1080)

print(Image_load.mode) # 返回图像的模式    # RGBA

img = Image_load.getdata()    # <ImagingCore at 0x1a8d4186e90> 包含像素内容的一个对象，注意是逐行拼接而成

list = list[img]    # 将对象转换成Python 的 list对象


"""=========================转化numpy对象可避免调用Image类============================="""
img_array = np.asarray(Image_load)  # 转化为numpy对象
# [weight,height,channel]

plt.imshow(img_array)
plt.show()


"""=========================测试反色===================================="""
img_divert = np.array([img_array[:,:,2],img_array[:,:,1],img_array[:,:,0]]).transpose(1,2,0)
# 此处注意更换位置为 [weight, height, channel]

plt.imshow(img_divert)
plt.show()


"""========================更改透明度============================="""
Image_load = Image.open('E:\Pictures\lock_picture.png')
img_alpha = Image_load.convert('RGBA')          # RGBA模式转换
R,G,B,A = img_alpha.split()             # RGBA分割

alpha = 0
# 此方法每个像素赋予相同透明度，输入参数为一固定值
img_alpha.putalpha(alpha)

# 每个像素随机赋予透明度
alpha = np.random.randint(0,255,[1080,1920])
# stack新建维度为第一维，故需要交换维度
img_alpha_correct = np.stack([np.array(R),np.array(G),np.array(B),alpha]).transpose(1,2,0)

plt.imshow(img_alpha_correct)
plt.show()