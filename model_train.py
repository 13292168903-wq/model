import copy
import time
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
import torch
import torch.nn as nn
import numpy as np
from model import LeNet

##数据加载
def train_val_data_process():
    train_data = FashionMNIST(root='./data', train=True,
                              transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor()]),
                              download=True)
    ##compose就是把后面的处理步骤按顺序组合起来 前面的resize就是调整尺寸 后面的好理解 就是转化成张量
    train_data,val_data = random_split(train_data,[round(len(train_data)*0.8),round(len(train_data)*0.2)])
    ##把数据集分好

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True,num_workers=8)  #初始化
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True,num_workers=8)

    return train_dataloader,val_dataloader
def train_model_process(model,train_dataloader,val_dataloader):

    device = torch.device("mps" if torch.mps.is_available() else "cpu") ##设备加载

    optimizer = torch.optim.Adam(model.parameters(),lr=0.001) ##梯度下降法的优化器 学习率为0.001
    ##parameters是去除模型中所有可训练的权重和偏置
    criterion = nn.CrossEntropyLoss()  ##损失函数 在分类当中一般用交叉商损失 回归用均方误差

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())  ##把最好的模型参数保存副本 防止最佳参数的改变
                                                        ##model.state_dict() 是模型全部卷积核、权重和偏置组成的字典

    ##提高准确度
    best_acc = 0.0  ##初始化参数
    train_loss_all = [] ##训练集的loss列表
    val_loss_all = []   ##测试集loss列表
    train_acc_all = []  ##训练集精度列表
    val_acc_all = []    ##测试集精度列表

    since = time.time()  ##看一下每次训练的时间

    ##追踪训练过程
    num_epochs = 50
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1,num_epochs))
        print("-"*10)

        ##初始化参数
        train_loss = 0.0 ##训练集loss
        val_loss = 0.0 ##测试集loss
        train_corrects = 0.0  ##训练集精度
        val_corrects  = 0.0  ##测试集精度
        ##样本数量
        train_num = 0
        val_num = 0

        for step,(b_x,b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            ##吧模型切换到训练模式
            model.train()
            ##训练一批次的数据
            output = model(b_x)
            ##找最大概率的索引
            pre_label = torch.argmax(output,1)

            loss = criterion(output,b_y) ##计算损失函数
            optimizer.zero_grad() ##梯度初始化为零 每一批次的数据梯度单独计算
            loss.backward()## 反向传播
            optimizer.step()  ##根据反向传播结果进行梯度更新

            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_label == b_y)  ##每正确一个 +1
            train_num += b_x.size(0) ##获得训练的样本集的数量
