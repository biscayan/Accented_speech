import torch.nn as nn
import torch.nn.functional as F


class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()

        # input channels, output channels (The number of kernels), kernel size, stride, padding

        self.conv1 = nn.Conv2d(13, 16, 3, 1, 1) #16 20*20
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2) # kernel size, stride, padding = 0 (default) #16 10*10

        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1) #32 10*10
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2) #32 5*5

        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1) #64 5*5
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2) #64 2x2

        self.fc1 = nn.Linear(1600, 1600) # input features, output features
        self.fc2 = nn.Linear(1600, 1600)
        self.fc3 = nn.Linear(800, 5)
        self.fc_out = nn.Linear(1600, 5)

    def forward(self, x):

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(-1, 1600)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = self.fc_out(x)

        return x