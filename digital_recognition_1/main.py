#手写数字识别
import model
import torch

import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import datasets
import model

#自己定义的利用训练好的模型进行测试
def show():

    #不使用梯度
    with torch.no_grad():

        img_path = './digital_recognition/pictures/test.png'
        image_origin = Image.open(img_path)

        #变成灰度图
        image_processed = image_origin.convert("L")

        image_test = datasets.test_transform(image_processed)
        image_test = image_test.unsqueeze(dim=0)    #在第0个位置加一个维度, 长度为1
        #print(image_test.shape)
        #设定模型为评估模式
        model.net.eval()
        #得到输出
        label_test = model.net(image_test)
        _,predicted = torch.max(label_test,1)
        print(predicted)

show()

epochs = 10
i = 0
while i<epochs:
    model.train(i)
    i+=1

show()
model.eval_3()
