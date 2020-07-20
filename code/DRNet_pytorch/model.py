import torch
import torch.nn as nn
import torch.nn.functional as F

from guided_filter.guided_filter import GuidedFilter as GuidedFilter

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
            seq['relu'] = nn.ReLU()
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
            seq['relu'] = nn.ReLU()
        self.seq = nn.Sequential(seq)

    def forward(self, x):
        out = self.seq(x)
        if self.kernel % 2 == 1:
            out = F.pad(out, (0, 1, 0, 1))
        return out


# Residual block, but with possibility of squeezing the channels
# @TODO Read a bit more about this to understand why this is usefull
class BottleneckBlock(torch.nn.Module):
    def __init__(self, channels_in, channels_out, kernel, stride, dilation=1, bias=False):
        super().__init__()

        self.conv1 = ConvBlock(channels_in, channels_out, kernel, stride, dilation, bias=bias)
        self.conv2 = ConvBlock(channels_out, channels_out, kernel, 1, dilation, bias=bias, relu=False)

        self.conv_squeeze = None
        if not (channels_in == channels_out and stride == 1):
            self.conv_squeeze = ConvBlock(channels_in, channels_out, 1, stride, dilation=1, bias=bias, relu=False)

        self.activate = nn.ReLU()

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)

        res = input
        if self.conv_squeeze is not None:
            res = self.conv_squeeze(res)

        output = output + res

        return self.activate(output)

class DRNetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        channels = 64

        self.layers = nn.ModuleList([])
        # Level 0
        self.layers.append(ConvBlock(3, channels, kernel=7, stride=1, dilation=1))

        # Level 1
        self.layers.append(BottleneckBlock(channels, channels, kernel=3, stride=2, dilation=1))
        self.layers.append(BottleneckBlock(channels, channels, kernel=3, stride=1, dilation=1))

        # Level 2
        self.layers.append(BottleneckBlock(channels, channels, kernel=3, stride=2, dilation=1))
        self.layers.append(BottleneckBlock(channels, channels, kernel=3, stride=1, dilation=1))

        # Level 3
        self.layers.append(BottleneckBlock(channels, channels, kernel=3, stride=1, dilation=2))
        self.layers.append(BottleneckBlock(channels, channels, kernel=3, stride=1, dilation=2))

        # Level 4
        self.layers.append(BottleneckBlock(channels, channels, kernel=3, stride=1, dilation=4))
        self.layers.append(BottleneckBlock(channels, channels, kernel=3, stride=1, dilation=4))

        # Level 5
        self.layers.append(BottleneckBlock(channels, channels, kernel=3, stride=1, dilation=2))
        self.layers.append(BottleneckBlock(channels, channels, kernel=3, stride=1, dilation=2))

        # Level 6
        self.layers.append(BottleneckBlock(channels, channels, kernel=3, stride=1, dilation=1))
        self.layers.append(BottleneckBlock(channels, channels, kernel=3, stride=1, dilation=1))

        # Level 7
        self.layers.append(BottleneckBlock(channels, channels, kernel=3, stride=1, dilation=1))
        self.layers.append(BottleneckBlock(channels, channels, kernel=3, stride=1, dilation=1))

        # Level 8
        self.layers.append(nn.Sequential(
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        ))

        self.layers.append(nn.Sequential(
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        ))

        # DECODER
        # Level 10
        self.layers.append(DeconvBlock(channels, channels, kernel=4, stride=2)) #i=19
        self.layers.append(nn.Sequential(
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        ))

        # Level 11
        self.layers.append(DeconvBlock(channels, channels, kernel=4, stride=2))
        self.layers.append(nn.Sequential(
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        ))

        self.pred_prob_temp = ConvBlock(channels, 1, kernel=1, stride=1, dilation=1, bias=True, bn=False, relu=False)
        self.guided_prob_temp = ConvBlock(channels, 1, kernel=3, stride=1, dilation=1, bias=True, bn=False, relu=False)

        self.guided_filter = GuidedFilter(4, eps=1e-2)

    def forward(self, input):
        output = input
        #print("in", input.shape)
        for i, layer in enumerate(self.layers):
            output = layer(output)
            #print("{}".format(i), output.shape)

        pred_prob_temp = self.pred_prob_temp(output)
        guided_prob_temp = self.guided_prob_temp(output)
        output = self.guided_filter(guided_prob_temp, pred_prob_temp)
        return output
