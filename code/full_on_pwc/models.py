import torch.nn as nn
import torch

from guided_filter.guided_filter import GuidedFilter as GuidedFilter

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from DDFlow_pytorch.model import Extractor, PWCNet
from FlowNetPytorch.models.FlowNetS import FlowNetS

from DDFlow_pytorch.model_utils import backwarp
try:
    from .correlation import correlation # the custom cost volume layer
except:
    sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python


###############################
######## BEST SO FAR ##########
###############################
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


############################
# ------- BASE MODEL ----- #
############################
class V3Adapt(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer6 = nn.Sequential(
            DeconvBlock(channels6, channels6, kernel=4, stride=2),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer5 = nn.Sequential(
            DeconvBlock(channels6+channels5, channels5, kernel=4, stride=2),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer4 = nn.Sequential(
            DeconvBlock(channels5+channels4, channels4, kernel=4, stride=2),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1)
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2):
        features = features1

        output = self.process_layer6(features[5])
        output = self.upscale_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = self.upscale_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = self.upscale_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density


#############################
# -- ADAPT WITH DILATION -- #
#############################
class V3Dilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer6 = nn.Sequential(
            DeconvBlock(channels6, channels6, kernel=4, stride=2),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            DeconvBlock(channels6+channels5, channels5, kernel=4, stride=2),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer4 = nn.Sequential(
            DeconvBlock(channels5+channels4, channels4, kernel=4, stride=2),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1)
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2):
        features = features1

        output = self.process_layer6(features[5])
        output = self.upscale_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = self.upscale_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = self.upscale_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density

#############################
# -- ADAPT WITH DILATION FULLL!! -- #
#############################
class V32Dilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer6 = nn.Sequential(
            DeconvBlock(channels6, channels6, kernel=4, stride=2),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            DeconvBlock(channels6+channels5, channels5, kernel=4, stride=2),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer4 = nn.Sequential(
            DeconvBlock(channels5+channels4, channels4, kernel=4, stride=2),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2):
        features = features1

        output = self.process_layer6(features[5])
        output = self.upscale_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = self.upscale_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = self.upscale_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density


class V33Dilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer6 = nn.Sequential(
            DeconvBlock(channels6, channels6, kernel=4, stride=2),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            DeconvBlock(channels6+channels5, channels5, kernel=4, stride=2),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer4 = nn.Sequential(
            DeconvBlock(channels5+channels4, channels4, kernel=4, stride=2),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2):
        features = features1

        output = self.process_layer6(features[5])
        output = self.upscale_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = self.upscale_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = self.upscale_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density

class V332Dilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            ConvBlock(channels6+channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer4 = nn.Sequential(
            ConvBlock(channels5+channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2):
        features = features1

        output = self.process_layer6(features[5])
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density

class V333Dilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            ConvBlock(channels6+channels5, channels6+channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6+channels5, channels6+channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6+channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer4 = nn.Sequential(
            ConvBlock(channels5+channels4, channels5+channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5+channels4, channels5+channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5+channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels4+channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4+channels3, channels4+channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4+channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2):
        features = features1

        output = self.process_layer6(features[5])
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density


class V34Dilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            ConvBlock(channels6+channels5, channels6+channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6+channels5, channels6+channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6+channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer4 = nn.Sequential(
            ConvBlock(channels5+channels4, channels5+channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5+channels4, channels5+channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5+channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels4+channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4+channels3, channels4+channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4+channels3, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2):
        features = features1

        output = self.process_layer6(features[5])
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density

class V341Dilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            ConvBlock(channels6+channels5, channels6+channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6+channels5, channels6+channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6+channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer4 = nn.Sequential(
            ConvBlock(channels5+channels4, channels5+channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5+channels4, channels5+channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5+channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels4+channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4+channels3, channels4+channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4+channels3, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale2_norm_layer = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1)
        )

        self.upscale1_norm_layer = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1)
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2):
        features = features1

        output = self.process_layer6(features[5])
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)

        output = F.interpolate(input=output, size=(features[1].shape[2], features[1].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale2_norm_layer(output)

        output = F.interpolate(input=output, size=(features[0].shape[2], features[0].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale1_norm_layer(output)

        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density


class V35Dilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2
        channels3 = 64

        self.after_upscale = nn.Sequential(
            ConvBlock(channels3, channels3*2, kernel=3, stride=1, dilation=1)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels3, channels3*2, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale2_norm_layer = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1)
        )

        self.upscale1_norm_layer = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1)
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2):
        features = features1

        # output = F.interpolate(input=features[3], size=(features[2].shape[2], features[2].shape[3]),
        #                        mode='bilinear', align_corners=False)
        # output = self.after_upscale(output)
        # output = torch.cat((features[2], output), 1)
        output = features[2]
        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density

class V351Dilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2
        channels3 = 64

        self.after_upscale = nn.Sequential(
            ConvBlock(channels3, channels3*2, kernel=3, stride=1, dilation=1)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels3, channels3*2, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale2_norm_layer = nn.Sequential(
            ConvBlock(channels3, 32, kernel=3, stride=1, dilation=1)
        )

        self.upscale1_norm_layer = nn.Sequential(
            ConvBlock(32, 16, kernel=3, stride=1, dilation=1)
        )

        self.output_layer = nn.Conv2d(16, 1, kernel_size=1)


    def cc_forward(self, features1, features2):
        features = features1

        output = features[2]
        output = self.process_all(output)
        output = F.interpolate(input=output, size=(features[1].shape[2], features[1].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale2_norm_layer(output)

        output = F.interpolate(input=output, size=(features[0].shape[2], features[0].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale1_norm_layer(output)

        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density



class V3EndFlow(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        corr_channels = 81

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(channels6*2, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer6 = nn.Sequential(
            DeconvBlock(channels6, channels6, kernel=4, stride=2),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(2*channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer5 = nn.Sequential(
            DeconvBlock(channels6+channels5, channels5, kernel=4, stride=2),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(2*channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer4 = nn.Sequential(
            DeconvBlock(channels5+channels4, channels4, kernel=4, stride=2),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(2*channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1)
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2, flow):
        features = features1

        # @TODO: Different weights to the flow per layer
        flow5 = F.interpolate(input=flow, size=(features2[5].shape[2], features2[5].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[5], tenFlow=flow5)
        output = self.process_layer6(torch.cat([features[5], back], 1))
        output = self.upscale_layer6(output)

        flow4 = F.interpolate(input=flow, size=(features2[4].shape[2], features2[4].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[4], tenFlow=flow4)
        output = torch.cat((output, self.process_layer5(torch.cat([features[4], back], 1))), 1)
        output = self.upscale_layer5(output)

        flow3 = F.interpolate(input=flow, size=(features2[3].shape[2], features2[3].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[3], tenFlow=flow3)
        output = torch.cat((output, self.process_layer4(torch.cat([features[3], back], 1))), 1)
        output = self.upscale_layer4(output)

        flow2 = F.interpolate(input=flow, size=(features2[2].shape[2], features2[2].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[2], tenFlow=flow2)
        output = torch.cat((output, self.process_layer3(torch.cat([features[2], back], 1))), 1)
        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw)
        return flow_fw, flow_bw, density


class V3EndFlowDilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        corr_channels = 81
        dilation = 2

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(channels6*2, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer6 = nn.Sequential(
            DeconvBlock(channels6, channels6, kernel=4, stride=2),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(2*channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            DeconvBlock(channels6+channels5, channels5, kernel=4, stride=2),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(2*channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer4 = nn.Sequential(
            DeconvBlock(channels5+channels4, channels4, kernel=4, stride=2),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(2*channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1)
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2, flow):
        features = features1
        flow = flow.detach()

        # @TODO: Different weights to the flow per layer
        flow5 = F.interpolate(input=flow, size=(features2[5].shape[2], features2[5].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[5], tenFlow=flow5)
        output = self.process_layer6(torch.cat([features[5], back], 1))
        output = self.upscale_layer6(output)

        flow4 = F.interpolate(input=flow, size=(features2[4].shape[2], features2[4].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[4], tenFlow=flow4)
        output = torch.cat((output, self.process_layer5(torch.cat([features[4], back], 1))), 1)
        output = self.upscale_layer5(output)

        flow3 = F.interpolate(input=flow, size=(features2[3].shape[2], features2[3].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[3], tenFlow=flow3)
        output = torch.cat((output, self.process_layer4(torch.cat([features[3], back], 1))), 1)
        output = self.upscale_layer4(output)

        flow2 = F.interpolate(input=flow, size=(features2[2].shape[2], features2[2].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[2], tenFlow=flow2)
        output = torch.cat((output, self.process_layer3(torch.cat([features[2], back], 1))), 1)
        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw)
        return flow_fw, flow_bw, density


class V32EndFlowDilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        corr_channels = 81
        dilation = 2

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(channels6*2, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer6 = nn.Sequential(
            DeconvBlock(channels6, channels6, kernel=4, stride=2),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(2*channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            DeconvBlock(channels6+channels5, channels5, kernel=4, stride=2),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(2*channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer4 = nn.Sequential(
            DeconvBlock(channels5+channels4, channels4, kernel=4, stride=2),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(2*channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1)
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2, flow):
        features = features1
        flow = flow.detach()

        # @TODO: Different weights to the flow per layer
        flow5 = F.interpolate(input=flow, size=(features2[5].shape[2], features2[5].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[5], tenFlow=flow5)
        output = self.process_layer6(torch.cat([features[5], back], 1))
        output = self.upscale_layer6(output)

        flow4 = F.interpolate(input=flow, size=(features2[4].shape[2], features2[4].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[4], tenFlow=flow4)
        output = torch.cat((output, self.process_layer5(torch.cat([features[4], back], 1))), 1)
        output = self.upscale_layer5(output)

        flow3 = F.interpolate(input=flow, size=(features2[3].shape[2], features2[3].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[3], tenFlow=flow3)
        output = torch.cat((output, self.process_layer4(torch.cat([features[3], back], 1))), 1)
        output = self.upscale_layer4(output)

        flow2 = F.interpolate(input=flow, size=(features2[2].shape[2], features2[2].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[2], tenFlow=flow2)
        output = torch.cat((output, self.process_layer3(torch.cat([features[2], back], 1))), 1)
        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw)
        return flow_fw, flow_bw, density


class V33EndFlowDilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(2*channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer6 = nn.Sequential(
            DeconvBlock(channels6, channels6, kernel=4, stride=2),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(2*channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            DeconvBlock(channels6+channels5, channels5, kernel=4, stride=2),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(2*channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer4 = nn.Sequential(
            DeconvBlock(channels5+channels4, channels4, kernel=4, stride=2),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(2*channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2, flow):
        features = features1
        flow = flow.detach()

        # @TODO: Different weights to the flow per layer
        flow5 = F.interpolate(input=flow, size=(features2[5].shape[2], features2[5].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[5], tenFlow=flow5)
        output = self.process_layer6(torch.cat([features[5], back], 1))
        output = self.upscale_layer6(output)

        flow4 = F.interpolate(input=flow, size=(features2[4].shape[2], features2[4].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[4], tenFlow=flow4)
        output = torch.cat((output, self.process_layer5(torch.cat([features[4], back], 1))), 1)
        output = self.upscale_layer5(output)

        flow3 = F.interpolate(input=flow, size=(features2[3].shape[2], features2[3].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[3], tenFlow=flow3)
        output = torch.cat((output, self.process_layer4(torch.cat([features[3], back], 1))), 1)
        output = self.upscale_layer4(output)

        flow2 = F.interpolate(input=flow, size=(features2[2].shape[2], features2[2].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[2], tenFlow=flow2)
        output = torch.cat((output, self.process_layer3(torch.cat([features[2], back], 1))), 1)
        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw)
        return flow_fw, flow_bw, density


class V332EndFlowDilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(2*channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(2*channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            ConvBlock(channels6+channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(2*channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer4 = nn.Sequential(
            ConvBlock(channels5+channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(2*channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2, flow):
        features = features1
        flow = flow.detach()

        # @TODO: Different weights to the flow per layer
        flow5 = F.interpolate(input=flow, size=(features2[5].shape[2], features2[5].shape[3]),
                              mode='bicubic', align_corners=False) / 32
        back = backwarp(tenInput=features2[5], tenFlow=flow5)
        output = self.process_layer6(torch.cat([features[5], back], 1))
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)

        flow4 = F.interpolate(input=flow, size=(features2[4].shape[2], features2[4].shape[3]),
                              mode='bicubic', align_corners=False) / 16
        back = backwarp(tenInput=features2[4], tenFlow=flow4)
        output = torch.cat((output, self.process_layer5(torch.cat([features[4], back], 1))), 1)
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)

        flow3 = F.interpolate(input=flow, size=(features2[3].shape[2], features2[3].shape[3]),
                              mode='bicubic', align_corners=False) / 8
        back = backwarp(tenInput=features2[3], tenFlow=flow3)
        output = torch.cat((output, self.process_layer4(torch.cat([features[3], back], 1))), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)

        flow2 = F.interpolate(input=flow, size=(features2[2].shape[2], features2[2].shape[3]),
                              mode='bicubic', align_corners=False) / 4
        back = backwarp(tenInput=features2[2], tenFlow=flow2)
        output = torch.cat((output, self.process_layer3(torch.cat([features[2], back], 1))), 1)
        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw)
        return flow_fw, flow_bw, density

class V332SingleFlow(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 6
        channels6 = 196
        self.flow_layer6 = nn.Sequential(
            ConvBlock(2, channels6, kernel=3, stride=1, dilation=1)
        )
        self.process_layer6 = nn.Sequential(
            ConvBlock(2*channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        # Layer 5
        channels5 = 128
        self.flow_layer5 = nn.Sequential(
            ConvBlock(2, channels5, kernel=3, stride=1, dilation=1)
        )
        self.process_layer5 = nn.Sequential(
            ConvBlock(2*channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            ConvBlock(channels6+channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        # Layer 4
        channels4 = 96
        self.flow_layer4 = nn.Sequential(
            ConvBlock(2, channels4, kernel=3, stride=1, dilation=1)
        )
        self.process_layer4 = nn.Sequential(
            ConvBlock(2*channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer4 = nn.Sequential(
            ConvBlock(channels5+channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1)
        )

        # Layer 3
        channels3 = 64
        self.flow_layer3 = nn.Sequential(
            ConvBlock(2, channels3, kernel=3, stride=1, dilation=1)
        )
        self.process_layer3 = nn.Sequential(
            ConvBlock(2*channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2, flow):
        features = features1
        flow = flow.detach()

        # @TODO: Different weights to the flow per layer
        flow5 = F.interpolate(input=flow, size=(features2[5].shape[2], features2[5].shape[3]),
                              mode='bicubic', align_corners=False) / 32
        # back = backwarp(tenInput=features2[5], tenFlow=flow5)
        output = self.process_layer6(torch.cat([features[5], self.flow_layer6(flow5)], 1))
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)

        flow4 = F.interpolate(input=flow, size=(features2[4].shape[2], features2[4].shape[3]),
                              mode='bicubic', align_corners=False) / 16
        # back = backwarp(tenInput=features2[4], tenFlow=flow4)
        output = torch.cat((output, self.process_layer5(torch.cat([features[4], self.flow_layer5(flow4)], 1))), 1)
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)

        flow3 = F.interpolate(input=flow, size=(features2[3].shape[2], features2[3].shape[3]),
                              mode='bicubic', align_corners=False) / 8
        # back = backwarp(tenInput=features2[3], tenFlow=flow3)
        output = torch.cat((output, self.process_layer4(torch.cat([features[3], self.flow_layer4(flow3)], 1))), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)

        flow2 = F.interpolate(input=flow, size=(features2[2].shape[2], features2[2].shape[3]),
                              mode='bicubic', align_corners=False) / 4
        # back = backwarp(tenInput=features2[2], tenFlow=flow2)
        output = torch.cat((output, self.process_layer3(torch.cat([features[2], self.flow_layer3(flow2)], 1))), 1)
        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw)
        return flow_fw, flow_bw, density


class V34EndFlowDilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(2*channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer6 = nn.Sequential(
            DeconvBlock(channels6, channels6, kernel=4, stride=2),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(2*channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            DeconvBlock(channels6+channels5, channels5, kernel=4, stride=2),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(2*channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer4 = nn.Sequential(
            DeconvBlock(channels5+channels4, channels4, kernel=4, stride=2),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1)
        )

        # Layer 3
        channels3 = 64

        self.process_all = nn.Sequential(
            ConvBlock(channels4, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2, flow):
        features = features1
        flow = flow.detach()

        # @TODO: Different weights to the flow per layer
        flow5 = F.interpolate(input=flow, size=(features2[5].shape[2], features2[5].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[5], tenFlow=flow5)
        output = self.process_layer6(torch.cat([features[5], back], 1))
        output = self.upscale_layer6(output)

        flow4 = F.interpolate(input=flow, size=(features2[4].shape[2], features2[4].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[4], tenFlow=flow4)
        output = torch.cat((output, self.process_layer5(torch.cat([features[4], back], 1))), 1)
        output = self.upscale_layer5(output)

        flow3 = F.interpolate(input=flow, size=(features2[3].shape[2], features2[3].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[3], tenFlow=flow3)
        output = torch.cat((output, self.process_layer4(torch.cat([features[3], back], 1))), 1)
        output = self.upscale_layer4(output)

        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw)
        return flow_fw, flow_bw, density


class V35EndFlowDilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(2*channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer6 = nn.Sequential(
            DeconvBlock(channels6, channels6, kernel=4, stride=2),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(2*channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            DeconvBlock(channels6+channels5, channels5, kernel=4, stride=2),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(2*channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer4 = nn.Sequential(
            DeconvBlock(channels5+channels4, channels4, kernel=4, stride=2),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(2*channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2, flow):
        features = features1
        flow = flow.detach()

        # @TODO: Different weights to the flow per layer
        flow5 = F.interpolate(input=flow, size=(features2[5].shape[2], features2[5].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[5], tenFlow=flow5)
        output = self.process_layer6(torch.cat([features[5], back], 1))
        output = self.upscale_layer6(output)

        flow4 = F.interpolate(input=flow, size=(features2[4].shape[2], features2[4].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[4], tenFlow=flow4)
        output = torch.cat((output, self.process_layer5(torch.cat([features[4], back], 1))), 1)
        output = self.upscale_layer5(output)

        flow3 = F.interpolate(input=flow, size=(features2[3].shape[2], features2[3].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[3], tenFlow=flow3)
        output = torch.cat((output, self.process_layer4(torch.cat([features[3], back], 1))), 1)
        output = self.upscale_layer4(output)

        flow2 = F.interpolate(input=flow, size=(features2[2].shape[2], features2[2].shape[3]),
                              mode='bicubic', align_corners=False)
        back = backwarp(tenInput=features2[2], tenFlow=flow2)
        output = torch.cat((output, self.process_layer3(torch.cat([features[2], back], 1))), 1)
        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw)
        return flow_fw, flow_bw, density


class Baseline1(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.flownet = FlowNetS(retCat2=True)

        if load_pretrained:
            path = '../FlowNetPytorch/flownets_bn_EPE2.459.pth.tar'
            self.flownet.load_state_dict(torch.load(path)['state_dict'])

        self.output_layer = nn.Conv2d(194, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def flow_to_orig(self, flow, orig_shape):
        flow_shape = flow.shape
        flow = torch.nn.functional.interpolate(input=flow, size=(orig_shape[2], orig_shape[3]),
                                                      mode='bicubic', align_corners=False)

        # Resize weights when rescaling to original size
        flow[:, 0, :, :] *= float(orig_shape[2]) / float(flow_shape[2])
        flow[:, 1, :, :] *= float(orig_shape[3]) / float(flow_shape[3])

        return flow

    def forward(self, frame1, frame2):
        flow_fw, combined = self.flownet(torch.cat([frame1, frame2], 1))
        flow_bw, _ = self.flownet(torch.cat([frame2, frame1], 1))

        flow_fw = self.flow_to_orig(flow_fw, frame1.shape)
        flow_bw = self.flow_to_orig(flow_bw, frame1.shape)

        density = self.output_layer(combined)
        return flow_fw, flow_bw, density


class V3Correlation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        corr_channels = 81

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(corr_channels + channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer6 = nn.Sequential(
            DeconvBlock(channels6, channels6, kernel=4, stride=2),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(corr_channels + channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer5 = nn.Sequential(
            DeconvBlock(channels6+channels5, channels5, kernel=4, stride=2),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(corr_channels + channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer4 = nn.Sequential(
            DeconvBlock(channels5+channels4, channels4, kernel=4, stride=2),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(corr_channels + channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1)
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2):
        features = features1

        corr = correlation.FunctionCorrelation(tenFirst=features1[5], tenSecond=features2[5])
        output = self.process_layer6(torch.cat([features[5], corr], 1))
        output = self.upscale_layer6(output)

        corr = correlation.FunctionCorrelation(tenFirst=features1[4], tenSecond=features2[4])
        output = torch.cat((output, self.process_layer5(torch.cat([features[4], corr], 1))), 1)
        output = self.upscale_layer5(output)

        corr = correlation.FunctionCorrelation(tenFirst=features1[3], tenSecond=features2[3])
        output = torch.cat((output, self.process_layer4(torch.cat([features[3], corr], 1))), 1)
        output = self.upscale_layer4(output)

        corr = correlation.FunctionCorrelation(tenFirst=features1[2], tenSecond=features2[2])
        output = torch.cat((output, self.process_layer3(torch.cat([features[2], corr], 1))), 1)
        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density


class V5Dilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        self.output_layer6 = nn.Conv2d(channels6, 1, kernel_size=1)

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            ConvBlock(channels6+channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        self.output_layer5 = nn.Conv2d(channels5, 1, kernel_size=1)

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer4 = nn.Sequential(
            ConvBlock(channels5+channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1)
        )

        self.output_layer4 = nn.Conv2d(channels4, 1, kernel_size=1)

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2):
        features = features1

        output = self.process_layer6(features[5])
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)
        return self.output_layer(output)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density