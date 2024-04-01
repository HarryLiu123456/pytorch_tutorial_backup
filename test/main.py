import torch
import datasets
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import numpy
import torchvision

if __name__ == '__main__':

    net = torch.load('./test/model.pth',map_location=torch.device('cpu'))
    device = torch.device('cpu')
    
    #测一批数据
    def eval_1():
        net.eval()
        dataiter = iter(datasets.test_loader)
        images ,labels = dataiter.__next__()
        
        outputs = net(images)
        _,predicted = torch.max(outputs,1)
        
        #输出标签
        print('labels: ', ' '.join('%5s' % datasets.classes[labels[j]] for j in range(4) ) )
        #输出预测结果
        print('Predicted: ', ' '.join('%5s' % datasets.classes[predicted[j]] for j in range(4) ) )
        
    #测一测总正确率
    def eval_2():
        net.eval()
        correct = 0
        total = 0
        
        for data in datasets.test_loader:
            images, labels = data
            
            outputs = net(images)
            _,predicted = torch.max(outputs,1)
            
            total+=labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print('Accuracy of the network on the 10000 test images: %d %%' % (correct*100/total))
    
    def show():
        net.eval()
        img_path = './test/picture/test1.jpg'
        image_origin = Image.open(img_path)

        image_test = datasets.transform(image_origin)
        image_test = image_test.unsqueeze(dim=0)    #在第0个位置加一个维度, 长度为1
        #print(image_test.shape)
        net.eval()
        label_test = net(image_test)
        _,predicted = torch.max(label_test,1)
        print('Predicted: %5s' % (datasets.classes[predicted[0]]) )
        
        #展示图片
        def imshow(img):
            img = img / 2 + 0.5
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg,(1,2,0)))
            plt.show()
        imshow(torchvision.utils.make_grid(image_test))

    eval_1()
    eval_2()
    show()


