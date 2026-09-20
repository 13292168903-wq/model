from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from model import LeNet

##数据加载
def train_val_data_process():
    train_data = FashionMNIST(root='./data', train=True,
                              transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor()]),
                              download=True)
    train_data,val_data = Data.random_split(train_data,[round(len(train_data)*0.8),round(len(train_data)*0.2)])

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True,num_workers=8)
    val_dataloader = DataLoader(train_data, batch_size=64, shuffle=True,num_workers=8)

    return train_dataloader,val_dataloader