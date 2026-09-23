import copy
import time
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from model import MLP
import matplotlib.pyplot as plt
from plot import train_loader


##数据加载
def train_val_data_process():
    train_data = FashionMNIST(root='./data', train=True,
                              transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor()]),
                              download=True)
    ##compose就是把后面的处理步骤按顺序组合起来 前面的resize就是调整尺寸 后面的好理解 就是转化成张量
    train_data,val_data = random_split(train_data,[round(len(train_data)*0.8),round(len(train_data)*0.2)])
    ##把数据集分好

    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True,num_workers=0)  #初始化
    val_dataloader = DataLoader(val_data, batch_size=128, shuffle=True,num_workers=0)

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
    num_epochs = 20
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

        ##差不多的数据处理方式 我们对验证集进行前向传播 不进行反向传播
        for step,(b_x,b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)##放数据
            b_y = b_y.to(device)

            output = model(b_x)  ##前向传播
            pre_label = torch.argmax(output,1)
            loss = criterion(output,b_y)    ##数据处理

            val_loss += loss.item() * b_x.size(0)  ##收集测试集的精度和loss
            val_corrects += torch.sum(pre_label == b_y)
            val_num += b_x.size(0)

        ##训练已经结束 现在收集数据
        train_acc_all.append(train_corrects/train_num)  ##总的正确除以总数 下面全是这个结构
        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_corrects/val_num)

        print("{},train loss:{:.4f},train accuracy:{:.4f} validation loss:{:.4f},validation accuracy:{:.4f}".format(epoch,
               train_loss_all[-1],train_acc_all[-1], val_loss_all[-1],val_acc_all[-1])  )
        ##寻找最好参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_used = time.time() - since

        print("该轮训练用时：{:.0f}min{:.0f}s".format(time_used//60, time_used%60))
        torch.save(best_model_wts,"./best_MLP_model.pth")

        ##用pandas保存数据
    train_process = pd.DataFrame({"epoch":range(1,num_epochs+1),
                                    "Train_loss":train_loss_all,
                                    "Train_accuracy":[acc.item() for acc in train_acc_all],
                                    "Validation_loss":val_loss_all,
                                    "Validation_accuracy":[acc.item() for acc in val_acc_all]})
    return train_process

def maplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"],train_process["Train_loss"],label="Train Loss",color="blue")
    plt.plot(train_process["epoch"],train_process["Validation_loss"],label="Val Loss",color="red")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(1,2,2)
    plt.plot(train_process["epoch"],train_process["Train_accuracy"],label="Train Accuracy",color="blue")
    plt.plot(train_process["epoch"],train_process["Validation_accuracy"],label="Val Accuracy",color="red")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    MLP = MLP()
    train_loader,val_loader = train_val_data_process()
    train_process = train_model_process(MLP,train_loader,val_loader)
    maplot_acc_loss(train_process)



