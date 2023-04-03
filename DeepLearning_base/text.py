import torch
from torch import nn
import torch.nn.functional as F

x=torch.rand(1,5)
y=torch.Tensor([0,1,0,1,0]).reshape(1,5)
w1=torch.tensor(0.8,requires_grad=True)
w2=torch.tensor(0.8,requires_grad=True)
b=torch.tensor(0.2)
epochs=10000
LOSS_MSE=nn.MSELoss()
LOSS_MSE_absolute=[]
w_mse=[[],[]]
learn_rate=0.05
for epoch in range(epochs):
    Z = F.logsigmoid(w1 * x + b)
    L1=LOSS_MSE(Z,y)
    L1.backward()
    with torch.no_grad():
        w1d=w1.grad.detach()  # 拿出梯度值
        w_mse[0].append(w1d)
        w_mse[1].append(L1.detach().numpy())
        w1 -= learn_rate * w1.grad  # 更新参数
        LOSS_MSE_absolute.append((Z-y)[0][0].detach().numpy())  # y1-y1'
    w1.grad.zero_()  #
# 设置好requires_grad的值为True
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y1 = x ** 2
y2 = y1 * 2
y3 = y1 + y2

print(y1, y1.requires_grad)
print(y2, y2.requires_grad)
print(y3, y3.requires_grad)

y3.backward(torch.ones(y3.shape))  # y1.backward() y2.backward()
print(x.grad)

