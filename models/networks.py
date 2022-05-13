from audioop import bias
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

class Bottleneck(nn.Module):
    def __init__(self, Conv, input_channel, output_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        expansion = 4
        self.conv1 = Conv(input_channel, output_channel, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(output_channel)

        self.conv2 = Conv(output_channel, output_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)

        self.conv3 = Conv(output_channel, expansion * output_channel, kernel_size=1, stride=stride, padding=0)
        self.bn3 = nn.BatchNorm2d(expansion * output_channel)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if expansion != 1 or stride!=1:
            self.downsample = Conv(input_channel, expansion * output_channel, kernel_size=1, stride=stride, padding=0)
            self.bn_d = nn.BatchNorm2d(expansion * output_channel)

    def forward(self, x, temp, ticket):
        residual = x
        out = self.conv1(x, temp, ticket)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, temp, ticket)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out, temp, ticket)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual, temp, ticket)
            #TODO:確認　論文によると、以下の層は必要ないかも。
            residual = self.bn_d(residual)

        out = self.relu(out + residual)
        return out
 
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

# ResNet18(cifar10用)
class ResNet(MaskedNet):
    def __init__(self, num_class, mask_initial_value=0.):
        super(ResNet, self).__init__()

        Conv = functools.partial(SoftMaskedConv2d, mask_initial_value=mask_initial_value)

        self.conv0 = Conv(3, 16, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(16)

        self.stage1 = ResStage(Conv, 16, 16, 1)
        self.stage2 = ResStage(Conv, 16, 32, 2)
        self.stage3 = ResStage(Conv, 32, 64, 2)

        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64, num_class)
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

#　ImageNet用
class ResNet50(MaskedNet):
    def __init__(self, num_class=1000, mask_initial_value=0.):
        super(ResNet50, self).__init__()

        Conv = functools.partial(SoftMaskedConv2d, mask_initial_value=mask_initial_value)

        self.conv0 = Conv(3, 64, 7, 3, 2)
        self.bn0 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)

        self.bottleneckblock10 = Bottleneck(Conv, 64, 64)
        self.bottleneckblock11 = Bottleneck(Conv, 256, 64)
        self.bottleneckblock12 = Bottleneck(Conv, 256, 64)

        self.bottleneckblock20 = Bottleneck(Conv, 256, 128, stride=2)
        self.bottleneckblock21 = Bottleneck(Conv, 512, 128)
        self.bottleneckblock22 = Bottleneck(Conv, 512, 128)
        self.bottleneckblock23 = Bottleneck(Conv, 512, 128)

        self.bottleneckblock30 = Bottleneck(Conv, 512, 256, stride=2)
        self.bottleneckblock31 = Bottleneck(Conv, 1024, 256)
        self.bottleneckblock32 = Bottleneck(Conv, 1024, 256)
        self.bottleneckblock33 = Bottleneck(Conv, 1024, 256)
        self.bottleneckblock34 = Bottleneck(Conv, 1024, 256)
        self.bottleneckblock35 = Bottleneck(Conv, 1024, 256)
        
        self.bottleneckblock40 = Bottleneck(Conv, 1024, 512, stride=2)
        self.bottleneckblock41 = Bottleneck(Conv, 2048, 512)
        self.bottleneckblock42 = Bottleneck(Conv, 2048, 512)

        self.avgpool = nn.AvgPool2d(7,stride=1)
        self.classifier = nn.Linear(2048, num_class)
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
        out = self.maxpool(out)
        out = self.bottleneckblock10(out, self.temp, self.ticket)
        out = self.bottleneckblock11(out, self.temp, self.ticket)
        out = self.bottleneckblock12(out, self.temp, self.ticket)
        out = self.bottleneckblock20(out, self.temp, self.ticket)
        out = self.bottleneckblock21(out, self.temp, self.ticket)
        out = self.bottleneckblock22(out, self.temp, self.ticket)
        out = self.bottleneckblock23(out, self.temp, self.ticket)
        out = self.bottleneckblock30(out, self.temp, self.ticket)
        out = self.bottleneckblock31(out, self.temp, self.ticket)
        out = self.bottleneckblock32(out, self.temp, self.ticket)
        out = self.bottleneckblock33(out, self.temp, self.ticket)
        out = self.bottleneckblock34(out, self.temp, self.ticket)
        out = self.bottleneckblock35(out, self.temp, self.ticket)
        out = self.bottleneckblock40(out, self.temp, self.ticket)
        out = self.bottleneckblock41(out, self.temp, self.ticket)
        out = self.bottleneckblock42(out, self.temp, self.ticket)
        out = self.avgpool(out)
        out = out.view(x.size(0), -1)
        out = self.classifier(out)
        return out
