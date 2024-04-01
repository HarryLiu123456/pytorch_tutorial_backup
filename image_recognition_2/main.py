import datasets
import model
import mynet

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    #测试datasets
    '''
    #展示图片
    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.show()
    #随机抓取一些图片
    dataiter = iter(datasets.train_loader)
    images ,labels = dataiter.__next__()
    imshow(torchvision.utils.make_grid(images))
    print(''.join('%5s' % datasets.classes[labels[j]] for j in range(4) ) )
    '''
    
    #测试model
    model.eval_1()
    model.eval_2()
    model.train(10)
    model.eval_1()
    model.eval_2()
    
    torch.save(mynet.net, 'model.pth')
    
    
    
    
    
    
    