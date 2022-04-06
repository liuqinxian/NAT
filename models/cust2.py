import torch
import torch.nn as nn

class cust2(nn.Module):
    def __init__(self, args):
        super(cust2, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3)
        # self.conv2 = nn.Conv3d(32, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.AvgPool3d(kernel_size=3, stride=3)

        self.conv3 = nn.Conv3d(32, 32, kernel_size=3)
        # self.conv4 = nn.Conv3d(64, 32, kernel_size=3)
        self.conv5 = nn.Conv3d(32, 1, kernel_size=3)
        self.bn2 = nn.BatchNorm3d(1)
        self.pool2 = nn.AvgPool3d(kernel_size=3, stride=3)
        
        self.line1 = nn.Linear(729, 32)
        self.line2 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        # x = self.bn1(x)
        x = self.pool1(x)
        
        # x = self.dropout(x)

        x = self.relu(self.conv3(x))
        # x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        # x = self.bn2(x)
        x = self.pool2(x)
        
        # x = self.dropout(x)

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        # print(x.shape)
        x = self.relu(self.line1(x))
        # x = self.line2(x)
        x = self.sigmoid(self.line2(x))
        return x



