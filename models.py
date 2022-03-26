import math

import torch
from torch import nn
import torch.nn.functional as F

from utils import weight_init

class BasicBlock(nn.Module):

    def __init__(self, channel):
        super(BasicBlock, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class MiniResnet18(nn.Module):
    """Backbone network

    MiniResnet18 architecture is a mini imitation of resnet18.

    """

    def __init__(self, out_features=2):
        """inilization

        Args:
            out_features: default 2, which is easy to be visualized

        """
        super(MiniResnet18, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.block1 = BasicBlock(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.block2 = BasicBlock(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.block3 = BasicBlock(128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, out_features)
    
    def forward(self, x):
        """module forward

        Args:
            x: input tensor of images, which shape is (batch size, 28, 28 , 1)
        
        Returns:
            A Tensor which shape is (batch_size, out_features)

        """
        x = self.conv1(x)
        x = self.block1(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.block2(x)
        x = self.pooling(x)
        x = self.conv3(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class LeNet_pp(nn.Module):
    """Backbone network

    LeNet++ architecture which is come from the paper CenterLoss published in ECCV 2016. This CNNs Architectures is sample. Visit the paper CenterLoss Table 1 for more details.

    """

    def __init__(self, out_features=2):
        """inilization

        Args:
            out_features: default 2, which is easy to be visualized

        """
        super(LeNet_pp, self).__init__()
        self.name = 'LeNet_pp'
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.features = nn.Sequential(
            nn.Linear(128*3*3, out_features),
            nn.PReLU()
        )
        self.apply(weight_init)

    def forward(self, x):
        """module forward

        Args:
            x: input tensor of images, which shape is (batch size, 28, 28 , 1)
        
        Returns:
            A Tensor which shape is (batch_size, out_features)

        """
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = self.features(x)
        return x


class AngleLinear(nn.Module):
    """The AngleLinear layer for Face Loss

    This class is a simple implementation of the combine of L-softmax/A-softmax/CosFace/ArcFace. The formula can be seen in README.md

    """

    def __init__(self, in_features=2, out_features=10, w_norm=True, x_norm=True, s=1, m1=1.0, m2=0.5, m3=0.0, device='cuda'):
        """inilization

        Args:
            in_features: the dimension of the input features
            out_features: the dimension of the output features
            w_norm: if we do normilization on the weight
            x_norm: if we do normilization on the input features
            s: scale
            m1/m2/m3: parameters of angle
            device: which device should we put the tensor on
        """
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m1 = m1
        self.m2 = m2
        if self.m2 != 0:
            self.m2_cos = math.cos(m2)
            self.m2_sin = math.sin(m2)
        self.m3 = m3
        self.w_norm = w_norm
        self.x_norm = x_norm
        self.device = device
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.m1_lambda = [
            lambda x: x**0,
            lambda x: x,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, x, label=None):
        """module forward

        Args:
            x: input tensor of features, which shape is (batch size, in_features)
            lable: input tensor of lables, which shape is (batch size, 1)

        Returns:
            A Tensor which shape is (batch_size, out_features)
        """
        if self.x_norm:
            x = F.normalize(x)
        if self.w_norm:
            W = F.normalize(self.weight)
        else:
            W = self.weight
        output = F.linear(x, W)
        if label is not None and not (self.m1 == 1 and self.m2 == 0 and self.m3 == 0):
            if not self.x_norm:
                x = F.normalize(x)
            if not self.w_norm:
                W = F.normalize(self.weight)
            cosine = F.linear(x, W)
            cosine[cosine>0.99] = 0.99
            cosine[cosine<-0.99] = -0.99
            assert self.m1 <= 5
            phi =self.m1_lambda[self.m1](cosine)
            if self.m2 != 0:
                phi_sine = torch.sqrt(1.0 - torch.pow(phi, 2))
                phi = self.m2_cos * phi - self.m2_sin * phi_sine
            if self.m3 != 0:
                phi = phi - self.m3
            # easy margin
            phi = torch.where(cosine > 0, phi, cosine)
            one_hot = torch.zeros(cosine.size(), device=self.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            if not self.x_norm:
                output *= x.pow(2).sum(1,keepdim=True).sqrt()
            if not self.w_norm:
                output *= self.weight.pow(2).sum(1,keepdim=True).sqrt()
        output *= self.s
        return output