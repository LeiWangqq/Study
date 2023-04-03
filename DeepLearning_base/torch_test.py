"""
@FileName：torch_test.py\n
@Description：\n
@Author：Wang.Lei\n
@Time：2022/9/29 10:53\n
@Department：Postgrate\n
"""
import torch

a = torch.rand(5,3,requires_grad=True)
b = torch.rand(5,3,requires_grad=True)
print(a.is_leaf)   # 判断是否为叶子结点，此处为True
loss = a + b**2

loss.backward(torch.ones(loss.size()))       # gradient形状与y一致，注意此处不是标量就必须确定形状
print(loss.is_leaf)     # 判断是否为叶子结点，此处为 False, 不是叶子结点，故梯度计算后被释放


x = torch.ones(1)
b = torch.rand(1, requires_grad = True)
w = torch.rand(1, requires_grad = True)
y = w * x # 等价于y=w.mul(x)
z = y + b # 等价于z=y.add(b)

"""=====================控制台命令及输出结果========================"""

x.is_leaf, w.is_leaf, b.is_leaf
# (True, True, True)
y.is_leaf, z.is_leaf
# (False, False)

z.grad_fn
# grad_fn可以查看这个variable的反向传播函数，
# z是add函数的输出，所以它的反向传播函数是AddBackward

z.grad_fn.next_functions
# next_functions保存grad_fn的输入，是一个tuple，tuple的元素也是Function
# 第一个是y，它是乘法(mul)的输出，所以对应的反向传播函数y.grad_fn是MulBackward
# 第二个是b，它是叶子节点，由用户创建，grad_fn为None

z.grad_fn.next_functions[0][0] == y.grad_fn
# True
# variable的grad_fn对应着和图中的function相对应


y.grad_fn.next_functions
# ((<AccumulateGrad at 0x7f60b09c2898>, 0), (None, 0))
# 第一个是w，叶子节点，需要求导，梯度是累加的
# 第二个是x，叶子节点，不需要求导，所以为None


# 为了能够多次反向传播需要指定retain_graph来保留这些buffer,即计算图
