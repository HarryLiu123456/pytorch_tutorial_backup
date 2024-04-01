import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #输入图片为3通道，输出为6通道，卷积核大小5X5
        self.conv1 = nn.Conv2d(3,6,5)
        #在第一个项目中pool采用函数方式进行，这里采用模块方式，因为没有pool操作没有参数，不需要训练，所以两者结果一样
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
       
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #view函数来把x的0维之外的维度铺平，这里好像就省略了计算后面的维度数的函数
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#创建Net的实例
net = Net()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
net.to(device)  #把模型加载到GPU上

print("param value size:", (net.state_dict()["conv2.bias"]).size())
print("param value size:", (net.state_dict()["conv2.weight"]).size())

