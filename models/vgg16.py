import torch
import torch.nn as nn
import torchvision.models as models

class vgg(nn.Module):
    def __init__(self, args=None):
        super(vgg, self).__init__()
        self.vgg16 = models.vgg16(pretrained=args) # pretrained = True
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1),
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
        x = self.vgg16(x)
        return x