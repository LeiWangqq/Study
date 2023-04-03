"""
@FileName：MNIST_test.py\n
@Description：\n
@Author：Wang.Lei\n
@Time：2022/10/11 19:21\n
@Department：Postgrate\n
"""
# 引入库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pathlib as Path
import torch.optim as optim

# -------------------------------创建data路径------------------------------
now_path = Path.Path.cwd()
data_path = now_path / "data"
Path.Path(data_path).mkdir(parents=True, exist_ok=True)
"""
    Path后跟上路径，mkdir中需要传入两个参数:
        parents：如果父目录不存在，是否创建父目录。
        exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常。
"""
# ---------------------------定义超参-------------------------------------------
Batch_size = 50
Epochs = 10
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #--------------------------定义pipeline变换数据--------------------------------
pipeline = transforms.Compose([
    transforms.ToTensor(),
    # 注意要使用中括号
    transforms.Normalize((0.1307,), (0.3081,))
])
# --------------------------读取并装载MNIST数据---------------------------------
train_set = datasets.MNIST(str(data_path), train=True, download=True, transform=pipeline)
test_set = datasets.MNIST(str(data_path), train=False, download=True, transform=pipeline)
train_dataloader = DataLoader(train_set, shuffle=True, batch_size=Batch_size)
test_dataloader = DataLoader(test_set, shuffle=True, batch_size=Batch_size)


# ---------------------------定义网络模型----------------------------------------
class mymnist(nn.Module):
    def __init__(self):
        super(mymnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 5, 3)

        self.fc1 = nn.Linear(500, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)  # size = 10*24*24
        x = F.max_pool2d(x, 2, 2)  # size = 10*12*12
        x = F.relu(x)

        x = self.conv2(x)  # size = 5*10*10
        x = F.sigmoid(x)

        x = torch.flatten(x, 1, end_dim=-1)  # 1*[5*10*10] 从索引为【1】的列拉伸为一维Tensor
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# --------------------------定义优化器-------------------------------------
mymodel = mymnist().to(Device)
optimizer = optim.Adam(mymodel.parameters())  # 放入模型参数



# -----------------------------定义训练模型---------------------------------------
def train_model(mymodel, device, train_loader, optimizer, epoch):
    # 模型训练
    mymodel.train()
    for batch_index, (data, target) in enumerate(train_loader):
        # 部署到device
        data, target = data.to(device), target.to(device)
        # 梯度初始化归零
        optimizer.zero_grad()
        # 训练后结果
        output = mymodel(data)
        # 计算损失
        loss = F.cross_entropy(output, target)
        # 找到概率最大值下标
        pred = output.max(1, keepdim=True)  # 1 表示横轴
        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()
        if batch_index % 3000 == 0:
            print('train epoch ： {}\t  Loss:{:.6f}'.format(epoch, loss.item()))

# -------------------------------定义测试模型---------------------------------------
def test_model(model, device, test_loader):
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    with torch.no_grad():  # 不会计算梯度和反向传播
        # 部署到device上
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试数据
            test_loss += F.cross_entropy(output, target).item()
            # 找到概率值最大的索引
            pred = torch.max(output, dim=1)[1]
            # 累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('Test: Average Loss ：{:.6f}, Accuracy : {:.3f}'.format(test_loss,
                                                                     100.0 * correct / len(test_loader.dataset)))


for epoch in range(1, Epochs + 1):
    train_model(mymodel, Device, train_dataloader, optimizer, epoch)
    test_model(mymodel, Device, test_dataloader)
