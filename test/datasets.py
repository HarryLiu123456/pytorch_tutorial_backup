import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

transform=transforms.Compose([        
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                ])

train_dataset=datasets.CIFAR10(
                    root='./image_recognition/data',  
                    train=True,     
                    download=False,  
                    transform=transform
                )
train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=4,  
                    shuffle=True,
                    num_workers = 2
                )
test_dataset=datasets.CIFAR10(
                    root='./image_recognition/data',  
                    train=False,     
                    download=False,  
                    transform=transform
                )
test_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=4,  
                    shuffle=True,    
                    num_workers = 2
                )

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')



