#开始训练
import torch.optim as optim
import datasets
import mynet
import torch.nn as nn
import torch

import torchvision
import matplotlib.pyplot as plt
import numpy as np

net = mynet.Net()

def train(epoch):
    optimizer = optim.SGD(net.parameters(),lr=0.01)     #lr代表学习率 
    criterion = nn.CrossEntropyLoss()
    net.train()   #设置为训练模式
    running_loss = 0.0
    for i,data in enumerate(datasets.train_loader):
        #得到输入和标签
        inputs, labels = data   #data是一个包含两个张量的元组，使用这句话可以将两个张量分开
        #消除梯度
        optimizer.zero_grad()
        #前向传播、计算损失、反向传播、更新参数
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        #####print("labels:{}".format(labels))   #########
        #####print("outputs:{}".format(outputs))  #########
        loss.backward()
        optimizer.step()
        #打印日志
        running_loss += loss.item()
        '''
            #每一百个输出一次
        if i % 100 == 0:
            print('[%d,%5d]loss:%.3f' % (epoch+1,i+1,running_loss/100)
                  )
            running_loss = 0.0
        '''
    #每次迭代输出一次
    print('epoch:%d loss:%.3f' % (epoch+1,running_loss*64/60000) )
    running_loss = 0.0

def eval_1():
    correct = 0
    total = 0
    with torch.no_grad():   #或者model.eval()
        for data in datasets.test_loader:
            images,labels = data
            outputs = net(images)
            #传入了两个参数一个是数据，另一个表示在哪个维度操作，显然第一个维度长度是64即64个数据标签，第二维为长度为10的预测数组
            #_,是占位符，max函数返回两个值，一个是包含最大值的张量（即最大的概率），第二个是包含对应张量的索引（即对应数字），！！！注意
            #此时返回的张量仍是二维，第一维长度为64
            _,predicted = torch.max(outputs.data,1)
            total+=labels.size(0)   #size（0）返回labels第一维的长度
            correct+=(predicted==labels).sum().item()   #比较的结果是产生一个布尔张量, item函数为取值但不返回梯度grad
        print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))  #100是因为结果为百分之多少，total代表总数据量10000

def eval_2():
    class_correct = list(0. for i in range (10)) #python推导式
    class_total = list(0. for i in range (10))
    classes = [i for i in range(10)]
    with torch.no_grad():
        for data in datasets.test_loader:
            images,labels = data
            outputs = net(images)
            _,predicted = torch.max(outputs,1)
            ######
            #print(labels)
            #print(predicted)
            #######
            c = (predicted==labels).squeeze()   #squeeze()用来压缩为内容数目为1的维度，测试了一下，这里好像没啥作用
            #######
            #print(c)
            #######
            for i in range(len(labels)):    #对labels逐个进行判断
                label = labels[i]        
                class_correct[label] +=c[i].item()
                class_total[label] +=1
    
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i],100*class_correct[i] / class_total[i]))

#输入十张图 输出十张图的图片及对应数字
def eval_3():
    with torch.no_grad():
        dataiter = iter(datasets.train_loader)
        images,labels = dataiter.__next__()
        image = images[:10]
        label = labels[:10]
        outputs = net(image)
        _,predicted = torch.max(outputs,1)
        #打印标签数组以及对应的预测值数组
        print(label)
        print(predicted)

    def imshow(img):
        img = img / 2 + 0.5 #逆归一化
        
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg,(1,2,0))) #transpose交换维度顺序为1 2 0，imshow绘制图像
        plt.show()                  #show展示绘制完的图像
    
    imshow(torchvision.utils.make_grid(image))



    
        
                
                
            
        

