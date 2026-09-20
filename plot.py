from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

train_data = FashionMNIST(root='./data', train=True,
                                     transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]),
                                     download=True)  ##下载数据

train_loader =DataLoader(train_data, batch_size=64, shuffle=True,num_workers=0)  ##处理数据

##获得batch的数据
for step,(b_x,b_y) in enumerate(train_loader):
    if step>0:
        break
batch_x = b_x.squeeze().numpy()
batch_y = b_y.numpy()
class_label = train_data.classes
print(class_label)


