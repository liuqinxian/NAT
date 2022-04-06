import torch
import torch.nn as nn

class cust4(nn.Module):
    def __init__(self):
        super(cust4, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv3d(1, 14, kernel_size=3)
        # self.conv2 = nn.Conv3d(16, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm3d(14)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=3)
        
        # self.conv3 = nn.Conv3d(32, 16, kernel_size=3)
        self.conv2 = nn.Conv3d(14, 1, kernel_size=3)
        self.bn2 = nn.BatchNorm3d(1)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=3)

        # self.conv3 = nn.Conv3d(14, 1, kernel_size=3)
        # self.bn3 = nn.BatchNorm3d(1)
        # self.pool3 = nn.MaxPool3d(kernel_size=3, stride=3)
        self.dropout = nn.Dropout(p=0.9)

        self.line = nn.Linear(729, 2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # x = self.conv3(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.dropout(x)

        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)
        # x = self.pool3(x)

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        # print(x.shape)
        x = self.line(x)
        x = self.sigmoid(x)
        return x



