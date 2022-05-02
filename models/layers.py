import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

def sigmoid(x):
    return float(1./(1.+np.exp(-x)))

class SoftMaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, mask_initial_value=0.):
        super(SoftMaskedConv2d, self).__init__()
        self.mask_initial_value = mask_initial_value
        
        self.in_channels = in_channels
        self.out_channels = out_channels    
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.xavier_normal_(self.weight)
        self.init_weight = nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)
        self.init_mask()
        
    def init_mask(self):
        self.mask_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        nn.init.constant_(self.mask_weight, self.mask_initial_value)

    def compute_mask(self, temp, ticket):
        scaling = 1. / sigmoid(self.mask_initial_value)
        if ticket: mask = (self.mask_weight > 0).float()
        else: mask = torch.sigmoid(temp * self.mask_weight)
        return scaling * mask
        
    def prune(self, temp):
        self.mask_weight.data = torch.clamp(temp * self.mask_weight.data, max=self.mask_initial_value)   

    def forward(self, x, temp=1, ticket=False):
        self.mask = self.compute_mask(temp, ticket)
        masked_weight = self.weight * self.mask
        out = F.conv2d(x, masked_weight, stride=self.stride, padding=self.padding)        
        return out
        
    def checkpoint(self):
        self.init_weight.data = self.weight.clone()       
        
    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()

    def extra_repr(self):
        return '{}, {}, kernel_size={}, stride={}, padding={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
