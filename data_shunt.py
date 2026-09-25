import torch
from model import LeNet
from mlp.model import MLP
from model_test import test_data_process

device = torch.device("mps" if torch.mps.is_available() else "cpu")
##加载模型
mlp = MLP()
mlp.load_state_dict(torch.load("./mlp/best_MLP_model.pth"))
mlp.to(device)

lenet = LeNet()
lenet.load_state_dict(torch.load("./best_model.pth"))
lenet.to(device)

##预测模式
lenet.eval()
mlp.eval()

test_dataloader = test_data_process(128)
threshold = 0.95

total_num = 0
correct_num = 0
handoff_num = 0

with torch.inference_mode():
    for x, y in test_dataloader:
        x = x.to(device)
        y = y.to(device)

        output = mlp(x)
        confidence, mlp_pred = torch.softmax(output, dim=1).max(dim=1)

        handoff = confidence < threshold
        final_pred = mlp_pred.clone()

        if handoff.any():
            final_pred[handoff] = lenet(x[handoff]).argmax(dim=1)

        total_num += x.size(0)
        correct_num += (final_pred == y).sum().item()
        handoff_num += handoff.sum().item()

print("测试集图片数：", total_num)
print("交给 LeNet 的图片数：", handoff_num)
print("转交比例：{:.2%}".format(handoff_num / total_num))
print("分流系统准确率：{:.2%}".format(correct_num / total_num))
