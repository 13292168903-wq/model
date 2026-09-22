import torch
import torch.nn as nn
from torchinfo import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(1, 6, 5,padding=2)  #卷积核
        self.sig = nn.ReLU()  #激活函数
        self.s2 = nn.MaxPool2d(2,2)  #平均池化
        self.c3 = nn.Conv2d(6, 16, 5)
        self.s4 = nn.MaxPool2d(2,2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.c1(x)
        x = self.sig(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.sig(x)
        x = self.s4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sig(x)
        x = self.fc2(x)
        x = self.sig(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = LeNet().to(device)
    print(summary(model,(1,1,28,28)))