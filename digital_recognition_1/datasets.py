#导入MNIST数据集
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torchvision
import matplotlib.pyplot as plt
import numpy as np

train_transform=transforms.Compose([        #数据预处理
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,),(0.3081,))
                ])

train_dataset=datasets.MNIST(
                    root='./digital_recognition/data',  #表示数据加载的相对目录
                    train=True,     #表示是否进行训练
                    download=True,  #表示是否自动下载
                    transform=train_transform
                )

#这个
train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=64,  #一批的数据量
                    shuffle=True    #是否打乱顺序
                )

test_transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,),(0.3081,))
                ])

test_dataset=datasets.MNIST(
                    root='./digital_recognition/data',
                    train=False,
                    transform=test_transform
                )

#这个
test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=64,
                    shuffle=True
                )

'''
def imshow(img):
    img = img / 2 + 0.5 #逆归一化
    
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0))) #transpose交换维度顺序为1 2 0，imshow绘制图像
    plt.show()                  #show展示绘制完的图像
    
#得到batch中的数据
dataiter = iter(train_loader)
images,labels = dataiter.__next__()
imshow(torchvision.utils.make_grid(images))
'''


