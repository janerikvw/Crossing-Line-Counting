import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class ConvBlock(torch.nn.Module):
    def __init__(self, channels_in, channels_out, kernel, stride, dilation=1, bias=False, bn=True, relu=True):
        super().__init__()

        seq = OrderedDict()
        seq['conv'] = nn.Conv2d(channels_in, channels_out, kernel, padding=int((kernel-1)/2*dilation),
                                stride=stride, bias=bias, dilation=dilation)
        if bn:
            seq['batch_norm'] = nn.BatchNorm2d(channels_out)
        if relu:
            seq['relu'] = nn.ReLU(inplace=True)
        self.seq = nn.Sequential(seq)

    def forward(self, x):
        out = self.seq(x)
        return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, channels_in, channels_out, kernel, stride, dilation=1, bias=False, bn=True, relu=True):
        super().__init__()

        self.kernel = kernel

        seq = OrderedDict()
        seq['deconv'] = nn.ConvTranspose2d(channels_in, channels_out, kernel, padding=int((kernel-1)/2*dilation), stride=stride, bias=bias, dilation=dilation)
        if bn:
            seq['batch_norm'] = nn.BatchNorm2d(channels_out)
        if relu:
            seq['relu'] = nn.ReLU(inplace=True)
        self.seq = nn.Sequential(seq)

    def forward(self, x):
        out = self.seq(x)
        if self.kernel % 2 == 1:
            out = F.pad(out, (0, 1, 0, 1))
        return out