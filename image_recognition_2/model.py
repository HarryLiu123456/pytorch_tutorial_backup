import datasets
import mynet

import torch.optim as optim
import torch.nn as nn
import torch

#定义损失函数
criterion = nn.CrossEntropyLoss()
#定义优化器
optimizer = optim.SGD(mynet.net.parameters(), lr = 0.001, momentum = 0.9)

def train(epoch):
    mynet.net.train()
    train_epoch = 1
    while train_epoch <= epoch:
        running_loss = 0.0
        for i,data in enumerate(datasets.train_loader,0):   #0代表起始位置
            inputs,labels = data
            inputs,labels = inputs.to(mynet.device), labels.to(mynet.device)    #把数据放到GPU/CPU上
            optimizer.zero_grad()
            outputs = mynet.net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print('epoch: %d loss: %.3f' % (train_epoch,running_loss / 50000))
        running_loss = 0.0
        train_epoch+=1
        
    print('Finish train.')

#调一批数据测一测
def eval_1():
    mynet.net.eval()
    dataiter = iter(datasets.test_loader)
    images ,labels = dataiter.__next__()
    images, labels = images.to(mynet.device), labels.to(mynet.device)    #把数据放到GPU/CPU上
    outputs = mynet.net(images)
    _,predicted = torch.max(outputs,1)
    #输出标签
    print('labels: ', ' '.join('%5s' % datasets.classes[labels[j]] for j in range(4) ) )
    #输出预测结果
    print('Predicted: ', ' '.join('%5s' % datasets.classes[predicted[j]] for j in range(4) ) )

#测一测总正确率
def eval_2():
    mynet.net.eval()
    correct = 0
    total = 0
    for data in datasets.test_loader:
        images, labels = data
        images, labels = images.to(mynet.device), labels.to(mynet.device)    #把数据放到GPU/CPU上
        outputs = mynet.net(images)
        _,predicted = torch.max(outputs,1)
        total+=labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (correct*100/total))
