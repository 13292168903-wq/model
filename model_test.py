import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet
from torch.utils.data import DataLoader,random_split

def test_data_process(batch_size=1):
    test_data = FashionMNIST(root='./data', train=False,
                              transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor()]),
                              download=True)
    ##compose就是把后面的处理步骤按顺序组合起来 前面的resize就是调整尺寸 后面的好理解 就是转化成张量


    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,num_workers=0)  #初始化


    return test_dataloader
def test_model_process(model,test_dataloader):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = model.to(device)

    test_correct= 0.0
    test_num = 0

    with torch.no_grad(): ##梯度值设为零 值进行前向传播 不进行梯度计算
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()
            output = model(test_data_x)
            per_lab = torch.argmax(output, dim=1)
            test_correct += torch.sum(per_lab == test_data_y).item()
            test_num += test_data_x.size(0)
    test_acc = test_correct/ test_num
    print("正确率为：",test_acc)


def test_model_process_xiangxi (model,test_dataloader):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = model.to(device)
    classes = FashionMNIST.classes
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)

            result_lab = pre_lab.item()
            test_lab = b_y.item()

            print("预测值：",classes[result_lab],"真实值：",classes[test_lab])

if __name__ == "__main__":
    model = LeNet()
    model.load_state_dict(torch.load('./best_model.pth'))
    test_dataloader = test_data_process()
    #test_model_process_xiangxi(model,test_dataloader)
    test_model_process(model,test_data_process())
