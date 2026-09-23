import torch
import torch.nn as nn
from torchinfo import summary

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = MLP().to(device)
    print(summary(model, (1, 1, 28, 28)))