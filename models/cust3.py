import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(block, self).__init__()
        self.bn = nn.BatchNorm3d(in_channel)
        self.relu = nn.ReLU()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=3, padding=1)
    def forward(self, x):
        # return self.conv(self.relu(self.bn(x)))
        return self.relu(self.bn(self.conv(x)))
    
class layer(nn.Module):
    def __init__(self, n_block, channels):
        super(layer, self).__init__()
        self.blocks = nn.ModuleList([
            block(channels[i], channels[i+1])
            for i in range(n_block)
        ])
        self.align = nn.Conv3d(channels[0], channels[n_block], kernel_size=1)
    def forward(self, x, skip=None):
        for ablock in self.blocks:
            x = ablock(x)
        return (x + self.align(skip))

class cust3(nn.Module):
    def __init__(self, args=None):
        super(cust3, self).__init__()
        # self.layer1 = layer(2, [1, 16, 32])
        # self.layer1 = layer(1, [1, 16])
        # self.layer1 = layer(3, [1, 16, 32, 64])
        self.layer1 = layer(2, [1, 16, 32])
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=3)

        # self.layer2 = layer(2, [32, 16, 1])
        # self.layer2 = layer(1, [16, 1])
        # self.layer2 = layer(3, [64, 32, 16, 1])
        self.layer2 = layer(2, [32, 32, 32])
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=3)

        self.layer3 = layer(2, [32, 16, 1])
        self.pool3 = nn.MaxPool3d(kernel_size=3, stride=3)

        # self.linear = nn.Linear(1000, 2)
        self.linear = nn.Linear(27, 2)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.type(torch.cuda.FloatTensor)

        skip = x
        x = self.layer1(x, skip)
        x = self.pool1(x)

        skip = x
        x = self.layer2(x, skip)
        x = self.pool2(x)

        skip = x
        x = self.layer3(x, skip)
        x = self.pool3(x)

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.linear(x)
        return x