import torch
import torch.nn as nn

class customize(nn.Module):
    def __init__(self, args):
        super(customize, self).__init__()
        self.conv1 = nn.Conv3d(1, 1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(1)
        self.pool1 = nn.AvgPool3d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv3d(1, 1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(1)
        self.pool2 = nn.AvgPool3d(kernel_size=3, stride=3)
        self.conv3 = nn.Conv3d(1, 1, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(1)
        self.pool3 = nn.MaxPool3d(kernel_size=3, stride=3)
        self.conv4 = nn.Conv3d(1, 1, kernel_size=3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.unsqueeze(1)  # [b, 1, 96, 96, 96]
        x = x.type(torch.cuda.FloatTensor)
        # print('1: ', x.shape)
        x = self.pool1(self.relu(self.bn1(x + self.conv1(x))))
        # print('2: ', x.shape)
        x = self.pool2(self.relu(self.bn2(x + self.conv2(x))))
        # print('3: ', x.shape)
        x = self.pool3(self.relu(self.bn3(x + self.conv3(x))))
        # print('4: ', x.shape)
        x = self.conv4(x)
        # print('5: ', x.shape)
        x = self.sigmoid(x)
        x = x.squeeze(4).squeeze(3).squeeze(2)
        return x


