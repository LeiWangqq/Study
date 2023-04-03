"""
@FileName：moco_res101.py\n
@Description：moco对权重文件进行预训练\n
@Author：Wang.Lei\n
@Time：2023/3/8 14:54\n
@Department：Postgrate\n
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchsummary import summary
import numpy as np
from PIL import Image
import os
import time
from collections import OrderedDict
import copy
from osgeo import gdal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# define transformation for query and key
# transformation for train
train_transform = transforms.Compose([
                transforms.Resize((250,250)),
                transforms.RandomResizedCrop(224),
                transforms.RandomApply([
                        transforms.ColorJitter(0.5, 0.5, 0.5)
                        ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
# define class to make query and key
class Split:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


path2data = './data'
os.makedirs(path2data, exist_ok=True)

# In[定义数据集]
import pathlib
def find_files(path, endwith='tif'):
    initial = []
    find_function = lambda path, endwith: pathlib.Path(path).rglob('*' + endwith)
    path_generator = find_function(path, endwith)
    for i in path_generator:
        initial.append(i)
    ans = initial
    return ans
pil_images = find_files(r'H:\daochu_classfied\3channel_tif_voc\MOCO')
pil_images_train = []
for i in pil_images:
    img_path = str(i)  # pathlib 路径转换字符串
    dataset = gdal.Open(img_path)  # gdal打开图像获取信息
    band = []
    for i in range(dataset.RasterCount):
        band.append(dataset.GetRasterBand(i + 1).ReadAsArray())
    BAND = np.array(band)
    BAND = BAND.transpose(1, 2, 0)  # 将图像从 CHW 转换为 HWC

    # 转换图片
    BAND = (BAND - BAND.min()) / (BAND.max() - BAND.min())  # 将图像转换为 0-1 的 float
    band = np.uint8(BAND * 255)  # 转换为 uint8
    im = Image.fromarray(band[:, :, 0:3])  # 输出前三通道
    im = Split(train_transform)(im)
    pil_images_train.append(im)



# define dataloader
train_dl = DataLoader(pil_images_train, batch_size=64, shuffle=True)

from resnet import ResNet
# I use q encoder as resnett18 model
# q_encoder = models.resnet101(pretrained=False)


q_encoder = ResNet(depth=101, in_channels=3,
                   stem_channels=None,
                   base_channels=64,
                   num_stages=4,
                   out_indices=(0, 1, 2, 3),
                   frozen_stages=1,
                   norm_cfg=dict(type='BN', requires_grad=True),
                   norm_eval=True,
                   style='pytorch',
                   )
# In[]
# define classifier for our task
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(q_encoder.fc.in_features, 100)),
    ('added_relu1', nn.ReLU()),
    ('fc2', nn.Linear(100, 50)),
    ('added_relu2', nn.ReLU()),
    ('fc3', nn.Linear(50, 25))
]))

# replace classifier
# and this classifier make representation have 25 dimention
q_encoder.fc = classifier

# In[]
# define encoder for key by coping q_encoder
k_encoder = copy.deepcopy(q_encoder)

# move encoders to device
q_encoder = q_encoder.to(device)
k_encoder = k_encoder.to(device)


# check model
summary(q_encoder, (3,224,224), device=device.type)


# define loss function
def loss_func(q,k,queue,t=0.05):
    # t: temperature

    N = q.shape[0] # batch_size
    C = q.shape[1] # channel

    # # bmm: batch matrix multiplication
    # pos = torch.exp(torch.div(torch.bmm(q.view(N,1,C), k.view(N,C,1)).view(N,1),t))
    # neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N,C),torch.t(queue)),t)),dim=1)
    # # denominator is sum over pos and neg
    # denominator = pos + neg
    # return torch.mean(-torch.log(torch.div(pos, denominator)))



    l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).view(N, 1)
    l_neg = torch.mm(q.view(N, C), torch.t(queue))
    # logits = torch.cat((l_pos, l_neg), 1)
    # label = torch.zeros(N)
    # label = label.type(torch.LongTensor)
    # loss = torch.nn.functional.cross_entropy(logits, label)

    logits = torch.cat([l_pos, l_neg], dim=1)

    # apply temperature
    logits /= t

    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss




# define optimizer
opt = optim.Adam(q_encoder.parameters())

# initialize the queue
queue = None
K = 8192  # K: number of negatives to store in queue

# fill the queue with negative samples
flag = 0
if queue is None:
    while True:
        with torch.no_grad():
            for img1, img2 in train_dl :
                # extract key samples
                # 输入为 N C H W
                xk = img1.to(device)
                k = k_encoder(xk).detach()

                if queue is None:
                    queue = k
                else:
                    if queue.shape[0] < K:  # queue < 8192
                        queue = torch.cat((queue, k), 0)
                    else:
                        flag = 1  # stop filling the queue

                if flag == 1:
                    break
        if flag == 1:
            break

queue = queue[:K]
print('number of negative samples in queue : ', len(queue))


def Training(q_encoder, k_encoder, num_epochs, queue=queue, loss_func=loss_func, opt=opt, data_dl=train_dl,
             sanity_check=False):
    loss_history = []
    momentum = 0.999
    start_time = time.time()
    path2weights = './models/q_weights.pt'
    len_data = len(data_dl.dataset)

    # q_encoder.train()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        q_encoder.train()
        running_loss = 0
        for img1, img2 in data_dl:
            # retrieve query and key
            xq = img1.to(device)
            xk = img2.to(device)

            # get model outputs
            q = q_encoder(xq)
            k = k_encoder(xk).detach()

            # normalize representations
            q = torch.div(q, torch.norm(q, dim=1).reshape(-1, 1))
            k = torch.div(k, torch.norm(k, dim=1).reshape(-1, 1))

            # get loss value
            loss = loss_func(q, k, queue)
            running_loss += loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            # update the queue
            queue = torch.cat((queue, k), 0)

            if queue.shape[0] > K:
                queue = queue[256:, :]

            # update k_encoder
            for q_params, k_params in zip(q_encoder.parameters(), k_encoder.parameters()):
                k_params.data.copy_(momentum * k_params + q_params * (1.0 - momentum))

        # store loss history
        epoch_loss = running_loss / len(data_dl.dataset)
        loss_history.append(epoch_loss)

        print('train loss: %.6f, time: %.4f min' % (epoch_loss, (time.time() - start_time) / 60))

        if sanity_check:
            break

    # save weights
    # torch.save(q_encoder.state_dict(), path2weights);
    return q_encoder, k_encoder, loss_history


# create folder to save q_encoder model weights
os.makedirs('./models', exist_ok=True)

# start training
num_epochs = 300
q_encoder, _, loss_history = Training(q_encoder, k_encoder, num_epochs=num_epochs, sanity_check=False)

