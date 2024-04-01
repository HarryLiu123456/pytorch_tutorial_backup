#构建神经网络
import torch.nn as nn
import torch.nn.functional as F #可以调用一些常见的函数，例如非线性和池化

import torch.optim as optim
import datasets

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #卷积层，输入图片为单通道输出为6通道，卷积核大小5*5，（卷积核系统给定了）
        self.conv1 = nn.Conv2d(1,6,5)
        #
        self.conv2 = nn.Conv2d(6,16,5)
        #把16X4X4的张量转成一个120维的张量（全连接层）
        self.fc1 = nn.Linear(16*4*4,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    
    def forward(self,x):
        #在（2，2）窗口上进行池化, relu激活让负数全为0
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)   #（2，2）可以直接写成2
        #将维度转成以batch为第一维，剩余维度相乘（铺平）为第二维
        #view函数中-1是让他根据后面的维度推断-1处的维度，不允许存在两个-1
        x = x.view(-1,self.num_flat_features(x))
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    #将2到n维铺平
    def num_flat_features(self,x):
        size = x.size()[1:] #第一维batch不考虑转换
        num_features = 1    #num_features存储后面的维度存储的数据长度之积，来把后面的数据铺平
        for s in size:
            num_features *=s
        return num_features

'''
net = Net()
print(net)
'''

'''
#向前传播
dataiter = iter(datasets.train_loader)
images,labels = dataiter.__next__()
image = images[:2]
label = labels[:2]
net = Net()

print(image.size())
print(label)
out = net(image)
print(out)

#计算损失
dataiter = iter(datasets.train_loader)
images,labels = dataiter.__next__()
image = images[:2]
label = labels[:2]
net = Net()

out = net(image)
criterion = nn.CrossEntropyLoss()
loss = criterion(out, label)
print(loss)
'''

'''
#反向传播与更新参数
#创建优化器
dataiter = iter(datasets.train_loader)
images,labels = dataiter.__next__()
image = images[:2]
label = labels[:2]
net = Net()

optimizer = optim.SGD(net.parameters(),lr=0.01)     #lr代表学习率 
criterion = nn.CrossEntropyLoss()
#在训练过程中
optimizer.zero_grad()   #消除梯度
out = net(image)
loss = criterion(out,label)
loss.backward()
optimizer.step()    #更新参数
'''