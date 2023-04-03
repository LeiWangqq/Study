# 导入库
import torch
from torch import nn,optim
import torchvision
from torchvision import datasets,transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号


# # 用于测试 MSE 平方误差对权重值的影响
# A=torch.linspace(0,1,1000)
# power=A**2*(1-A)
# plt.plot(A,power)
# plt.xlabel('绝对误差A：$|y_i-y_i^{‘}|$')
# plt.ylabel('$w_{11}$权重')
# plt.title('平方差MSE')
# plt.show()
#
#
# 用于测试交叉熵误差 CrossEntropy 对权重值的影响
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
        w1d=w1.grad.item()  # 拿出梯度值
        w_mse[0].append(w1d)
        w_mse[1].append(L1.detach().numpy())
        w1 -= learn_rate * w1.grad  # 更新参数
        LOSS_MSE_absolute.append((Z-y)[0][0].detach().numpy())  # y1-y1'
    w1.grad.zero_()  # 梯度清零，防止梯度累计


LOSS_CrossEntropy=nn.CrossEntropyLoss()
LOSS_CrossEntropy_absolute=[]
w_crossentropy=[[],[]]
for epoch in range(epochs):
    Z = F.logsigmoid(w2 * x + b)
    L2=LOSS_CrossEntropy(Z,y)
    w2.retain_grad()
    L2.backward()
    with torch.no_grad():
        w2d=w2.grad.item()  # 拿出梯度值
        w_crossentropy[0].append(w2d)
        w_crossentropy[1].append(L2.item())
        w2 -= learn_rate * w2.grad  # 更新参数
    LOSS_CrossEntropy_absolute.append((Z-y)[0][0].detach().numpy())  # y1-y1'
    w2.grad.zero_()  # 梯度清零，防止梯度累计


    with torch.no_grad():
        w2 -= learn_rate * w2.grad  # 更新参数
    # w2.grad.zero_()  # 梯度清零，防止梯度累计


w_m=plt.subplot(2,2,1)
w_m.plot(w_mse[0],w_mse[1])
plt.xlabel('gradient')
plt.ylabel('loss-MSE')
plt.subplots_adjust(wspace=1)

w_c=plt.subplot(2,2,2)
w_c.plot(w_crossentropy[0],w_crossentropy[1])
plt.xlabel('gradient')
plt.ylabel('loss-crossentropy')
plt.suptitle('w1梯度和损失函数之间关系')

w_m_a=plt.subplot(2,2,3)
w_m_a.plot(w_mse[0],LOSS_MSE_absolute)
plt.xlabel('gradient')
plt.ylabel('LOSS_MSE_absolute')
plt.subplots_adjust(wspace=1)


w_c_a=plt.subplot(2,2,4)
w_c_a.plot(w_mse[0],LOSS_CrossEntropy_absolute)
plt.xlabel('gradient')
plt.ylabel('LOSS_CrossEntropy_absolute')
plt.suptitle('w1梯度与损失函数-第一行\n 绝对误差-第二行关系')
plt.show()

# 测试detach方法
# 设置好requires_grad的值为True
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y1 = x ** 2

y2 = y1.detach() * 2     # 注意这里在计算y2的时候对y1进行了detach()
y3 = y1 + y2

print(y1, y1.requires_grad)
print(y2, y2.requires_grad)
print(y3, y3.requires_grad)


y3.backward(torch.ones(y3.shape))  # y1.backward() y2.backward()
print(x.grad)

