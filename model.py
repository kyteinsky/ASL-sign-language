import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 2)
        self.pool = nn.MaxPool2d(2)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, 3)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 512)
        self.batch_norm4 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 25)

    def forward(self, x):
        # correct shape
        x = x.reshape(-1, 1, 28, 28)
        # conv blocks
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batch_norm1(x)
        x = self.swish(self.conv2(x))
        x = self.dropout(x)
        x = self.batch_norm2(x)
        x = self.pool(self.swish(self.conv3(x)))
        x = self.dropout(x)
        x = self.batch_norm3(x)
        # reshape for linear layers
        x = x.view(-1, 128*4*4)
        # linear block
        x = self.swish(self.fc1(x))
        x = self.dropout(x)
        x = self.swish(self.fc2(x))
        x = self.dropout(x)
        x = self.batch_norm4(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        
    def swish(self, x):
        return x * torch.sigmoid(x)
