import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
import functools
from torch.nn import init
import copy

class MaskedNet(nn.Module):
    def __init__(self):
        super(MaskedNet, self).__init__()
        self.ticket = False

    def checkpoint(self):
        for m in self.mask_modules: m.checkpoint()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.checkpoint = copy.deepcopy(m.state_dict())

    def rewind_weights(self):
        for m in self.mask_modules: m.rewind_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.load_state_dict(m.checkpoint)
                
    def prune(self):
        for m in self.mask_modules: m.prune(self.temp)

class ResBlock(nn.Module):
    def __init__(self, Conv, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv_a = Conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn_a = nn.BatchNorm2d(out_channels)
        self.conv_b = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn_b = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x, temp, ticket):
        residual = x
        out = self.conv_a(x, temp, ticket)
        out = self.bn_a(out)
        out = F.relu(out, inplace=True)
        out = self.conv_b(out, temp, ticket)
        out = self.bn_b(out)
        if self.downsample is not None: residual = self.downsample(x)
        return F.relu(residual + out, inplace=True)
    
class ResStage(nn.Module):
    def __init__(self, Conv, in_channels, out_channels, stride=1):
        super(ResStage, self).__init__()
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Conv2d(in_channels, out_channels, 1, 2, 0, bias=False)
            
        self.block1 = ResBlock(Conv, in_channels, out_channels, stride, downsample)
        self.block2 = ResBlock(Conv, out_channels, out_channels)
        self.block3 = ResBlock(Conv, out_channels, out_channels)

    def forward(self, x, temp, ticket):
        out = self.block1(x, temp, ticket)
        out = self.block2(out, temp, ticket)
        out = self.block3(out, temp, ticket)
        return out

# ResNet18
class ResNet(MaskedNet):
    def __init__(self, mask_initial_value=0.):
        super(ResNet, self).__init__()

        Conv = functools.partial(SoftMaskedConv2d, mask_initial_value=mask_initial_value)

        self.conv0 = Conv(3, 16, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(16)

        self.stage1 = ResStage(Conv, 16, 16, 1)
        self.stage2 = ResStage(Conv, 16, 32, 2)
        self.stage3 = ResStage(Conv, 32, 64, 2)

        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64, 10)
        self.mask_modules = [m for m in self.modules() if type(m) == SoftMaskedConv2d]
        self.temp = 1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x, self.temp, self.ticket)), inplace=True)
        out = self.stage1(out, self.temp, self.ticket)
        out = self.stage2(out, self.temp, self.ticket)
        out = self.stage3(out, self.temp, self.ticket)
        out = self.avgpool(out)
        out = out.view(x.size(0), -1)
        out = self.classifier(out)
        return out

#TODO:ResNet50書く
