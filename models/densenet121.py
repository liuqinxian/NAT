import torch
import torch.nn as nn
import torchvision.models as models

class densenet121(nn.Module):
    def __init__(self, args=None):
        super(densenet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=args) # pretrained = True
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(1024, 1),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, 512),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(512, 64),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(64, 8),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.densenet121(x)
        return x