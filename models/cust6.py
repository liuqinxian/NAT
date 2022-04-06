import torch
import torch.nn as nn

from sklearn import OneHotEncoder


def CROP(x, rate):
    b, c, h, w, d = x.shape
    hh, ww, dd = h//rate, w//rate, d//rate
    h1, w1, d1 = (h-hh)//2, (w-ww)//2, (d-dd)//2
    h2, w2, d2 = h1+hh-1, w1+ww-1, d1+dd-1
    return x[:, :, h1:h2, w1:w2, d1:d2]


class POOL(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(POOL, self).__init__()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=3, stride=stride)
    
    def forward(self, x):
        return self.pool(self.relu(self.norm(self.conv(x))))


class Dense(nn.Module):
    def __init__(self, in_channel=1, growth_rate=2, stride=2):
        super(Dense, self).__init__()

        self.pool = POOL(in_channel, in_channel*growth_rate, stride)

        self.pool_p = POOL(in_channel*growth_rate, in_channel*growth_rate**2, stride)
        self.pool_c = POOL(in_channel*growth_rate, in_channel*growth_rate**2, stride)

        self.pool_f = POOL(in_channel*growth_rate**2, in_channel*growth_rate**3, stride)

        self.avg = nn.AvgPool3d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()

        self.rate = stride
    
    def forward(self, x):
        batch_size = x.shape[0]
        # x: 1, 96, 96, 96
        x = x.unsqueeze(1)

        # p: 2, 48, 48, 48
        p = self.pool(x)
        c = CROP(x, self.stride)

        # pp: 4, 24, 24, 24
        pp = self.pool_p(p+c)
        pc = CROP(p, self.stride)
        cp = self.pool_c(c)
        cc = CROP(c, self.stride)

        # y: 8, 12, 12, 12
        y = self.pool_f(pp+pc+cp+cc)

        # y: 8, 6, 6, 6
        y = self.avg(self.relu(y))
        y = y.reshape(batch_size, -1)

        return y

class Emb(nn.Module):
    def __init__(self, dim=32):
        super(Emb, self).__init__()
        self.l1 = nn.Linear()
        self.l2 = nn.Linear()
        self.l3 = nn.Linear()
        self.relu = nn.ReLU()
        
        self.enc = OneHotEncoder()
    def forward(self, x):
        x1 = x[:, 3:8]    # 数值型数据
        x1 =                     # 均一化
        x2 = x[:, 8:17]    # 类别型数据
        x2 =       
        

class Cust6(nn.Module):
    def __init__(self,):
        super(Cust6, self).__init__()

        self.dense = Dense()
        self.emb = Emb()
        self.fc = nn.Linear()
    
    def forward(self, x1, x2):
        y1 = self.dense(x1)
        y2 = self.emb(x2)
        y = torch.cat([y1, y2], dim=1)
        
         

