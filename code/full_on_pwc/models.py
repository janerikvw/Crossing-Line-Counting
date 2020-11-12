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
        ret6 = self.output_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)
        ret5 = self.output_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)
        ret4 = self.output_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)
        ret3 = self.output_layer(output)

        ret6 = F.interpolate(input=ret6, size=(features[2].shape[2], features[2].shape[3]),
                            mode='bilinear', align_corners=False)
        ret5 = F.interpolate(input=ret5, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret4 = F.interpolate(input=ret4, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret3 = F.interpolate(input=ret3, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)

        if self.training:
            ret = torch.cat((ret6, ret5, ret4, ret3), 1)
        else:
            ret = ret3
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density


class V501Dilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        self.output_layer5 = nn.Conv2d(channels5, 1, kernel_size=1)

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
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
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3 * 2, channels3 * 2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2):
        features = features1

        output = self.process_layer5(features[4])
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)
        ret5 = self.output_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)
        ret4 = self.output_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)
        ret3 = self.output_layer(output)

        ret5 = F.interpolate(input=ret5, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret4 = F.interpolate(input=ret4, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret3 = F.interpolate(input=ret3, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)

        if self.training:
            ret = torch.cat((ret5, ret4, ret3), 1)
        else:
            ret = ret3
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density


class V502Dilation(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        self.output_layer5 = nn.Conv2d(channels5, 1, kernel_size=1)

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
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
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels4+channels3, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3*2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3 * 2, channels3 * 2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3*2, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2):
        features = features1

        output = self.process_layer5(features[4])
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)
        ret5 = self.output_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)
        ret4 = self.output_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)
        ret3 = self.output_layer(output)

        ret5 = F.interpolate(input=ret5, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret4 = F.interpolate(input=ret4, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret3 = F.interpolate(input=ret3, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)

        if self.training:
            ret = torch.cat((ret5, ret4, ret3), 1)
        else:
            ret = ret3
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density


class V51Dilation(torch.nn.Module):
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
            ConvBlock(channels4 + channels3, channels3 * 2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3 * 2, channels3 * 2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3 * 2, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.output_layer41 = nn.Conv2d(channels3, 1, kernel_size=1)

        self.upscale_layer3 = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1)
        )

        self.output_layer3 = nn.Conv2d(channels3, 1, kernel_size=1)

        channels2 = 32
        self.process_layer2 = nn.Sequential(
            ConvBlock(channels2, channels2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels2, channels2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels2, channels2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels2, channels2, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer2 = nn.Sequential(
            ConvBlock(channels3 + channels2, channels2, kernel=3, stride=1, dilation=1),
            ConvBlock(channels2, channels2, kernel=3, stride=1, dilation=1)
        )

        self.output_layer2 = nn.Conv2d(channels2, 1, kernel_size=1)

    def cc_forward(self, features1, features2):
        features = features1

        output = self.process_layer6(features[5])
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)
        ret6 = self.output_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)
        ret5 = self.output_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)
        ret4 = self.output_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)
        ret41 = self.output_layer41(output)
        output = F.interpolate(input=output, size=(features[1].shape[2], features[1].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer3(output)
        ret3 = self.output_layer3(output)

        output = torch.cat((output, self.process_layer2(features[1])), 1)
        output = F.interpolate(input=output, size=(features[0].shape[2], features[0].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer2(output)
        ret2 = self.output_layer2(output)

        ret6 = F.interpolate(input=ret6, size=(ret2.shape[2], ret2.shape[3]),
                            mode='bilinear', align_corners=False)
        ret5 = F.interpolate(input=ret5, size=(ret2.shape[2], ret2.shape[3]),
                             mode='bilinear', align_corners=False)
        ret4 = F.interpolate(input=ret4, size=(ret2.shape[2], ret2.shape[3]),
                             mode='bilinear', align_corners=False)
        ret41 = F.interpolate(input=ret41, size=(ret2.shape[2], ret2.shape[3]),
                             mode='bilinear', align_corners=False)
        ret3 = F.interpolate(input=ret3, size=(ret2.shape[2], ret2.shape[3]),
                             mode='bilinear', align_corners=False)

        if self.training:
            ret = torch.cat((ret6, ret5, ret4, ret41, ret3, ret2), 1)
        else:
            ret = ret2
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density

class V52Dilation(torch.nn.Module):
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
            ConvBlock(channels4 + channels3, channels3 * 2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3 * 2, channels3 * 2, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3 * 2, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.output_layer41 = nn.Conv2d(channels3, 1, kernel_size=1)

        self.upscale_layer3 = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1)
        )

        self.output_layer3 = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2):
        features = features1

        output = self.process_layer6(features[5])
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)
        ret6 = self.output_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)
        ret5 = self.output_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)
        ret4 = self.output_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)
        ret41 = self.output_layer41(output)
        output = F.interpolate(input=output, size=(features[1].shape[2], features[1].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer3(output)
        ret3 = self.output_layer3(output)

        sizer = (ret3.shape[2], ret3.shape[3])

        ret6 = F.interpolate(input=ret6, size=sizer,
                            mode='bilinear', align_corners=False)
        ret5 = F.interpolate(input=ret5, size=sizer,
                             mode='bilinear', align_corners=False)
        ret4 = F.interpolate(input=ret4, size=sizer,
                             mode='bilinear', align_corners=False)
        ret41 = F.interpolate(input=ret41, size=sizer,
                             mode='bilinear', align_corners=False)

        if self.training:
            ret = torch.cat((ret6, ret5, ret4, ret41, ret3), 1)
        else:
            ret = ret3
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density



class V5Flow(torch.nn.Module):
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
        self.flow_layer5 = nn.Sequential(
            ConvBlock(2, channels5, kernel=3, stride=1, dilation=1)
        )

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
        self.flow_layer4 = nn.Sequential(
            ConvBlock(2, channels4, kernel=3, stride=1, dilation=1)
        )

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
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=dilation)
        )

        self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2, flow2):
        features = features1
        flow = flow2.clone().detach()

        output = self.process_layer6(features[5])
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)
        ret6 = self.output_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)
        ret5 = self.output_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)
        ret4 = self.output_layer4(output)

        flow2 = F.interpolate(input=flow, size=(features[2].shape[2], features[2].shape[3]),
                              mode='bilinear', align_corners=False)
        output = torch.cat((output, self.process_layer3(torch.cat([features[2], self.flow_layer3(flow2)], 1))), 1)
        output = self.process_all(output)
        ret3 = self.output_layer(output)

        ret6 = F.interpolate(input=ret6, size=(features[2].shape[2], features[2].shape[3]),
                            mode='bilinear', align_corners=False)
        ret5 = F.interpolate(input=ret5, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret4 = F.interpolate(input=ret4, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret3 = F.interpolate(input=ret3, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)

        if self.training:
            ret = torch.cat((ret6, ret5, ret4, ret3), 1)
        else:
            ret = ret3
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw)
        return flow_fw, flow_bw, density


class V5FlowFeatures(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

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
            ConvBlock(channels6+529, channels6, kernel=3, stride=1, dilation=dilation),
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
        self.flow_layer5 = nn.Sequential(
            ConvBlock(2, channels5, kernel=3, stride=1, dilation=1)
        )

        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5+661, channels5, kernel=3, stride=1, dilation=dilation),
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
        self.flow_layer4 = nn.Sequential(
            ConvBlock(2, channels4, kernel=3, stride=1, dilation=1)
        )

        self.process_layer4 = nn.Sequential(
            ConvBlock(channels4+629, channels4, kernel=3, stride=1, dilation=dilation),
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
        self.flow_layer3 = nn.Sequential(
            ConvBlock(2, channels3, kernel=3, stride=1, dilation=1)
        )

        self.process_layer3 = nn.Sequential(
            ConvBlock(channels3+597, channels3, kernel=3, stride=1, dilation=dilation),
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

    def cc_forward(self, features1, features2, flow2, flow_features):
        features = features1
        # flow = flow2.clone().detach()

        output = self.process_layer6(torch.cat((features[5], flow_features[0]), 1))
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)
        ret6 = self.output_layer6(output)

        output = torch.cat((output, self.process_layer5(torch.cat((features[4], flow_features[1]), 1))), 1)
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)
        ret5 = self.output_layer5(output)

        output = torch.cat((output, self.process_layer4(torch.cat((features[3], flow_features[2]), 1))), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)
        ret4 = self.output_layer4(output)

        # flow2 = F.interpolate(input=flow, size=(features[2].shape[2], features[2].shape[3]),
        #                       mode='bilinear', align_corners=False)
        # self.flow_layer3(flow2)
        output = torch.cat((output, self.process_layer3(torch.cat((features[2], flow_features[3]), 1))), 1)
        output = self.process_all(output)
        ret3 = self.output_layer(output)

        ret6 = F.interpolate(input=ret6, size=(features[2].shape[2], features[2].shape[3]),
                            mode='bilinear', align_corners=False)
        ret5 = F.interpolate(input=ret5, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret4 = F.interpolate(input=ret4, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret3 = F.interpolate(input=ret3, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)

        if self.training:
            ret = torch.cat((ret6, ret5, ret4, ret3), 1)
        else:
            ret = ret3
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class V51FlowFeatures(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

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
            ConvBlock(channels6+529, channels6, kernel=3, stride=1, dilation=dilation),
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
        self.flow_layer5 = nn.Sequential(
            ConvBlock(2, channels5, kernel=3, stride=1, dilation=1)
        )

        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5+661, channels5, kernel=3, stride=1, dilation=dilation),
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
        self.flow_layer4 = nn.Sequential(
            ConvBlock(2, channels4, kernel=3, stride=1, dilation=1)
        )

        self.process_layer4 = nn.Sequential(
            ConvBlock(channels4+629, channels4, kernel=3, stride=1, dilation=dilation),
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
        self.flow_layer3 = nn.Sequential(
            ConvBlock(2, channels3, kernel=3, stride=1, dilation=1)
        )

        self.process_layer3 = nn.Sequential(
            ConvBlock(channels3+597, channels3, kernel=3, stride=1, dilation=dilation),
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

    def cc_forward(self, features1, features2, flow2, flow_features):
        features = features1
        # flow = flow2.clone().detach()

        output = self.process_layer6(torch.cat((features[5], flow_features[0].clone().detach()), 1))
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)
        ret6 = self.output_layer6(output)

        output = torch.cat((output, self.process_layer5(torch.cat((features[4], flow_features[1].clone().detach()), 1))), 1)
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)
        ret5 = self.output_layer5(output)

        output = torch.cat((output, self.process_layer4(torch.cat((features[3], flow_features[2].clone().detach()), 1))), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)
        ret4 = self.output_layer4(output)

        # flow2 = F.interpolate(input=flow, size=(features[2].shape[2], features[2].shape[3]),
        #                       mode='bilinear', align_corners=False)
        # self.flow_layer3(flow2)
        output = torch.cat((output, self.process_layer3(torch.cat((features[2], flow_features[3].clone().detach()), 1))), 1)
        output = self.process_all(output)
        ret3 = self.output_layer(output)

        ret6 = F.interpolate(input=ret6, size=(features[2].shape[2], features[2].shape[3]),
                            mode='bilinear', align_corners=False)
        ret5 = F.interpolate(input=ret5, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret4 = F.interpolate(input=ret4, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret3 = F.interpolate(input=ret3, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)

        if self.training:
            ret = torch.cat((ret6, ret5, ret4, ret3), 1)
        else:
            ret = ret3
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

class V52FlowFeatures(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 5
        channels5 = 128
        self.flow_layer5 = nn.Sequential(
            ConvBlock(2, channels5, kernel=3, stride=1, dilation=1)
        )

        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5+661, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=dilation)
        )

        self.upscale_layer5 = nn.Sequential(
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1)
        )

        self.output_layer5 = nn.Conv2d(channels5, 1, kernel_size=1)

        # Layer 4
        channels4 = 96
        self.flow_layer4 = nn.Sequential(
            ConvBlock(2, channels4, kernel=3, stride=1, dilation=1)
        )

        self.process_layer4 = nn.Sequential(
            ConvBlock(channels4+629, channels4, kernel=3, stride=1, dilation=dilation),
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
        self.flow_layer3 = nn.Sequential(
            ConvBlock(2, channels3, kernel=3, stride=1, dilation=1)
        )

        self.process_layer3 = nn.Sequential(
            ConvBlock(channels3+597, channels3, kernel=3, stride=1, dilation=dilation),
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

    def cc_forward(self, features1, features2, flow2, flow_features):
        features = features1

        output = self.process_layer5(torch.cat((features[4], flow_features[1].clone().detach()), 1))
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)
        ret5 = self.output_layer5(output)

        output = torch.cat((output, self.process_layer4(torch.cat((features[3], flow_features[2].clone().detach()), 1))), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)
        ret4 = self.output_layer4(output)

        # flow2 = F.interpolate(input=flow, size=(features[2].shape[2], features[2].shape[3]),
        #                       mode='bilinear', align_corners=False)
        # self.flow_layer3(flow2)
        output = torch.cat((output, self.process_layer3(torch.cat((features[2], flow_features[3].clone().detach()), 1))), 1)
        output = self.process_all(output)
        ret3 = self.output_layer(output)

        ret6 = F.interpolate(input=ret6, size=(features[2].shape[2], features[2].shape[3]),
                            mode='bilinear', align_corners=False)
        ret5 = F.interpolate(input=ret5, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret4 = F.interpolate(input=ret4, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret3 = F.interpolate(input=ret3, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)

        if self.training:
            ret = torch.cat((ret6, ret5, ret4, ret3), 1)
        else:
            ret = ret3
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

class V5FlowWarping(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

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
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1)
        )

        self.output_layer6 = nn.Conv2d(channels6, 1, kernel_size=1)

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5*2, channels5, kernel=3, stride=1, dilation=dilation),
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
            ConvBlock(channels4*2, channels4, kernel=3, stride=1, dilation=dilation),
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
            ConvBlock(channels3*2, channels3, kernel=3, stride=1, dilation=dilation),
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

    def cc_forward(self, features1, features2, flow2, flow_features):
        features = features1
        flow = flow2.clone().detach()

        flow5 = F.interpolate(input=flow, size=(features2[5].shape[2], features2[5].shape[3]),
                              mode='bicubic', align_corners=False)
        back5 = backwarp(tenInput=features2[5], tenFlow=flow5)
        output = self.process_layer6(torch.cat((features[5], back5), 1))
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)
        ret6 = self.output_layer6(output)

        flow4 = F.interpolate(input=flow, size=(features2[4].shape[2], features2[4].shape[3]),
                              mode='bicubic', align_corners=False)
        back4 = backwarp(tenInput=features2[4], tenFlow=flow4)
        output = torch.cat((output, self.process_layer5(torch.cat((features[4], back4), 1))), 1)
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)
        ret5 = self.output_layer5(output)

        flow3 = F.interpolate(input=flow, size=(features2[3].shape[2], features2[3].shape[3]),
                              mode='bicubic', align_corners=False)
        back3 = backwarp(tenInput=features2[3], tenFlow=flow3)
        output = torch.cat((output, self.process_layer4(torch.cat((features[3], back3), 1))), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)
        ret4 = self.output_layer4(output)

        flow2 = F.interpolate(input=flow, size=(features2[2].shape[2], features2[2].shape[3]),
                              mode='bicubic', align_corners=False)
        back2 = backwarp(tenInput=features2[2], tenFlow=flow2)
        output = torch.cat((output, self.process_layer3(torch.cat((features[2], back2), 1))), 1)
        output = self.process_all(output)
        ret3 = self.output_layer(output)

        ret6 = F.interpolate(input=ret6, size=(features[2].shape[2], features[2].shape[3]),
                            mode='bilinear', align_corners=False)
        ret5 = F.interpolate(input=ret5, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret4 = F.interpolate(input=ret4, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret3 = F.interpolate(input=ret3, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)

        if self.training:
            ret = torch.cat((ret6, ret5, ret4, ret3), 1)
        else:
            ret = ret3
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

class V55FlowWarping(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(channels6*3, channels6, kernel=3, stride=1, dilation=dilation),
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
            ConvBlock(channels5*3, channels5, kernel=3, stride=1, dilation=dilation),
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
            ConvBlock(channels4*3, channels4, kernel=3, stride=1, dilation=dilation),
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
            ConvBlock(channels3*3, channels3, kernel=3, stride=1, dilation=dilation),
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

    def cc_forward(self, features1, features2, flow2, flow_features):
        features = features1
        flow = flow2.clone().detach()

        flow5 = F.interpolate(input=flow, size=(features2[5].shape[2], features2[5].shape[3]),
                              mode='bicubic', align_corners=False)
        back5 = backwarp(tenInput=features2[5], tenFlow=flow5)
        output = self.process_layer6(torch.cat((features[5], features2[5], back5), 1))
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)
        ret6 = self.output_layer6(output)

        flow4 = F.interpolate(input=flow, size=(features2[4].shape[2], features2[4].shape[3]),
                              mode='bicubic', align_corners=False)
        back4 = backwarp(tenInput=features2[4], tenFlow=flow4)
        output = torch.cat((output, self.process_layer5(torch.cat((features[4], features2[4], back4), 1))), 1)
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)
        ret5 = self.output_layer5(output)

        flow3 = F.interpolate(input=flow, size=(features2[3].shape[2], features2[3].shape[3]),
                              mode='bicubic', align_corners=False)
        back3 = backwarp(tenInput=features2[3], tenFlow=flow3)
        output = torch.cat((output, self.process_layer4(torch.cat((features[3], features2[3], back3), 1))), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)
        ret4 = self.output_layer4(output)

        flow2 = F.interpolate(input=flow, size=(features2[2].shape[2], features2[2].shape[3]),
                              mode='bicubic', align_corners=False)
        back2 = backwarp(tenInput=features2[2], tenFlow=flow2)
        output = torch.cat((output, self.process_layer3(torch.cat((features[2], features2[2], back2), 1))), 1)
        output = self.process_all(output)
        ret3 = self.output_layer(output)

        ret6 = F.interpolate(input=ret6, size=(features[2].shape[2], features[2].shape[3]),
                            mode='bilinear', align_corners=False)
        ret5 = F.interpolate(input=ret5, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret4 = F.interpolate(input=ret4, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret3 = F.interpolate(input=ret3, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)

        if self.training:
            ret = torch.cat((ret6, ret5, ret4, ret3), 1)
        else:
            ret = ret3
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class V51FlowWarping(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        dilation = 2

        # Layer 6
        channels6 = 196
        self.flow_layer6 = nn.Sequential(
            nn.Conv2d(in_channels=529, out_channels=2, kernel_size=3, stride=1,
                      padding=1)
        )

        self.process_layer6 = nn.Sequential(
            ConvBlock(channels6*2, channels6, kernel=3, stride=1, dilation=dilation),
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
        self.flow_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=661, out_channels=2, kernel_size=3, stride=1,
                      padding=1)
        )

        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5*2, channels5, kernel=3, stride=1, dilation=dilation),
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
        self.flow_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=629, out_channels=2, kernel_size=3, stride=1,
                      padding=1)
        )

        self.process_layer4 = nn.Sequential(
            ConvBlock(channels4*2, channels4, kernel=3, stride=1, dilation=dilation),
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
        self.flow_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=597, out_channels=2, kernel_size=3, stride=1,
                        padding=1)
        )

        self.process_layer3 = nn.Sequential(
            ConvBlock(channels3*2, channels3, kernel=3, stride=1, dilation=dilation),
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

    def cc_forward(self, features1, features2, flow2, flow_features):
        features = features1

        back6 = backwarp(tenInput=features2[5], tenFlow=self.flow_layer6(flow_features[0]))
        output = self.process_layer6(torch.cat((features[5], back6), 1))
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)
        ret6 = self.output_layer6(output)

        back5 = backwarp(tenInput=features2[4], tenFlow=self.flow_layer5(flow_features[1]))
        output = torch.cat((output, self.process_layer5(torch.cat((features[4], back5), 1))), 1)
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)
        ret5 = self.output_layer5(output)

        back4 = backwarp(tenInput=features2[3], tenFlow=self.flow_layer4(flow_features[2]))
        output = torch.cat((output, self.process_layer4(torch.cat((features[3], back4), 1))), 1)
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)
        ret4 = self.output_layer4(output)

        back3 = backwarp(tenInput=features2[2], tenFlow=self.flow_layer3(flow_features[3]))
        output = torch.cat((output, self.process_layer3(torch.cat((features[2], back3), 1))), 1)
        output = self.process_all(output)
        ret3 = self.output_layer(output)

        ret6 = F.interpolate(input=ret6, size=(features[2].shape[2], features[2].shape[3]),
                            mode='bilinear', align_corners=False)
        ret5 = F.interpolate(input=ret5, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret4 = F.interpolate(input=ret4, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret3 = F.interpolate(input=ret3, size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)

        if self.training:
            ret = torch.cat((ret6, ret5, ret4, ret3), 1)
        else:
            ret = ret3
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density



class Baseline2(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})


        # Layer 6
        self.flow_layer6 = nn.Sequential(
            nn.Conv2d(in_channels=529, out_channels=1, kernel_size=3, stride=1,
                      padding=1)
        )

        # Layer 5
        self.flow_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=661, out_channels=1, kernel_size=3, stride=1,
                      padding=1)
        )

        # Layer 4
        self.flow_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=629, out_channels=1, kernel_size=3, stride=1,
                      padding=1)
        )

        # Layer 3
        self.flow_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=597, out_channels=1, kernel_size=3, stride=1,
                      padding=1)
        )

    def cc_forward(self, features1, features2, flow2, flow_features):
        features = features1

        ret6 = F.interpolate(input=self.flow_layer6(flow_features[0]), size=(features[2].shape[2], features[2].shape[3]),
                            mode='bilinear', align_corners=False)
        ret5 = F.interpolate(input=self.flow_layer5(flow_features[1]), size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret4 = F.interpolate(input=self.flow_layer4(flow_features[2]), size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret3 = F.interpolate(input=self.flow_layer3(flow_features[3]), size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)

        if self.training:
            ret = torch.cat((ret6, ret5, ret4, ret3), 1)
        else:
            ret = ret3
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class Baseline21(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})


        # Layer 6
        self.flow_layer6 = nn.Sequential(
            nn.Conv2d(in_channels=529, out_channels=1, kernel_size=3, stride=1,
                      padding=1)
        )

        # Layer 5
        self.flow_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=661, out_channels=1, kernel_size=3, stride=1,
                      padding=1)
        )

        # Layer 4
        self.flow_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=629, out_channels=1, kernel_size=3, stride=1,
                      padding=1)
        )

        # Layer 3
        self.flow_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=597, out_channels=1, kernel_size=3, stride=1,
                      padding=1)
        )

    def cc_forward(self, features1, features2, flow2, flow_features):
        features = features1
        #
        # ret6 = F.interpolate(input=self.flow_layer6(flow_features[0]), size=(features[2].shape[2], features[2].shape[3]),
        #                     mode='bilinear', align_corners=False)
        # ret5 = F.interpolate(input=self.flow_layer5(flow_features[1]), size=(features[2].shape[2], features[2].shape[3]),
        #                      mode='bilinear', align_corners=False)
        # ret4 = F.interpolate(input=self.flow_layer4(flow_features[2]), size=(features[2].shape[2], features[2].shape[3]),
        #                      mode='bilinear', align_corners=False)
        ret3 = F.interpolate(input=self.flow_layer3(flow_features[3]), size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)

        ret = ret3
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class ResBlock(torch.nn.Module):
    def __init__(self, channels_in, channels_out, layers=3, dilation=1):
        super().__init__()

        self.in_channels = channels_in
        self.out_channels = channels_out

        self.shortcut = ConvBlock(channels_in=channels_in, channels_out=channels_out, kernel=3, stride=1, bias=False, bn=False, relu=False)
        self.activate = nn.ReLU(inplace=True)

        self.blocks = nn.Sequential(
            ConvBlock(channels_in, channels_out, kernel=3, stride=1, dilation=dilation),
            *[ConvBlock(channels_out, channels_out, kernel=3, stride=1, dilation=dilation) for _ in range(layers-2)],
            ConvBlock(channels_out, channels_out, kernel=3, stride=1, dilation=dilation, relu=False)
        )

    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut(): residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x


class V6Blocker(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

            # Layer 6
            channels6 = 196
            flow_channels6 = 529
            self.process_layer6 = ResBlock(channels6, channels6, layers=4, dilation=2)
            self.upscale_layer6 = ResBlock(channels6, channels6, layers=2, dilation=1)
            self.output_layer6 = nn.Conv2d(channels6, 1, kernel_size=1)

            # Layer 5
            channels5 = 128
            flow_channels5 = 661
            self.process_layer5 = ResBlock(channels5 + channels6, channels5, layers=4, dilation=2)
            self.upscale_layer5 = ResBlock(channels5, channels5, layers=2, dilation=1)
            self.output_layer5 = nn.Conv2d(channels5, 1, kernel_size=1)

            # Layer 4
            channels4 = 96
            flow_channels4 = 629
            self.process_layer4 = ResBlock(channels4 + channels5, channels4, layers=4, dilation=2)
            self.upscale_layer4 = ResBlock(channels4, channels4, layers=2, dilation=1)
            self.output_layer4 = nn.Conv2d(channels4, 1, kernel_size=1)

            # Layer 3
            channels3 = 64
            flow_channels3 = 597
            self.process_layer3 =  nn.Sequential(
                ResBlock(channels3 + channels4, channels3*2, layers=4, dilation=2),
                ResBlock(channels3*2, channels3, layers=3, dilation=2),
                ResBlock(channels3, channels3, layers=3, dilation=2)
            )

            self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2, flow2, flow_features):
        features = features1

        output = self.process_layer6(features[5])
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)
        ret6 = self.output_layer6(output)

        output = self.process_layer5(torch.cat((output, features[4]), 1))
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)
        ret5= self.output_layer5(output)

        output = self.process_layer4(torch.cat((output, features[3]), 1))
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)
        ret4 = self.output_layer4(output)

        output = self.process_layer3(torch.cat((output, features[2]), 1))
        ret3 = self.output_layer(output)

        ret6 = F.interpolate(input=ret6,
                             size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret5 = F.interpolate(input=ret5,
                             size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret4 = F.interpolate(input=ret4,
                             size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret3 = F.interpolate(input=ret3,
                             size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)

        if self.training:
            ret = torch.cat((ret6, ret5, ret4, ret3), 1)
        else:
            ret = ret3
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2,
                                                                                                ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class V61Blocker(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

            # Layer 6
            channels6 = 196
            flow_channels6 = 529
            self.process_layer6 = ResBlock(channels6, channels6, layers=4, dilation=2)
            self.upscale_layer6 = ResBlock(channels6, channels6, layers=2, dilation=1)
            self.output_layer6 = nn.Conv2d(channels6, 1, kernel_size=1)

            # Layer 5
            channels5 = 128
            flow_channels5 = 661
            self.process_layer5 = ResBlock(channels5 + channels6, channels5, layers=4, dilation=2)
            self.upscale_layer5 = ResBlock(channels5, channels5, layers=2, dilation=1)
            self.output_layer5 = nn.Conv2d(channels5, 1, kernel_size=1)

            # Layer 4
            channels4 = 96
            flow_channels4 = 629
            self.process_layer4 = ResBlock(channels4 + channels5, channels4, layers=4, dilation=2)
            self.upscale_layer4 = ResBlock(channels4, channels4, layers=2, dilation=1)
            self.output_layer4 = nn.Conv2d(channels4, 1, kernel_size=1)

            # Layer 3
            channels3 = 64
            flow_channels3 = 597
            self.process_layer3 =  ResBlock(channels3 + channels4, channels3, layers=4, dilation=2)
            self.output_layer3 = nn.Conv2d(channels3, 1, kernel_size=1)


            self.full_process = nn.Sequential(
                ResBlock(channels6 + channels5 + channels4 + channels3, channels3 * 4, layers=3, dilation=2),
                ResBlock(channels3 * 4, channels3 * 2, layers=3, dilation=2),
                ResBlock(channels3 * 2, channels3, layers=3, dilation=2)
            )

            self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2, flow2, flow_features):
        features = features1

        output = self.process_layer6(features[5])
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)
        result6 = output
        ret6 = self.output_layer6(output)

        output = self.process_layer5(torch.cat((output, features[4]), 1))
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)
        result5 = output
        ret5 = self.output_layer5(output)

        output = self.process_layer4(torch.cat((output, features[3]), 1))
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)
        result4 = output
        ret4 = self.output_layer4(output)

        output = self.process_layer3(torch.cat((output, features[2]), 1))
        ret3 = self.output_layer3(output)

        output_size = (output.shape[2], output.shape[3])
        output = torch.cat((F.interpolate(input=result6, size=output_size, mode='bilinear', align_corners=False),
                            F.interpolate(input=result5, size=output_size, mode='bilinear', align_corners=False),
                            F.interpolate(input=result4, size=output_size, mode='bilinear', align_corners=False),
                            output), 1)
        output = self.full_process(output)
        ret = self.output_layer(output)

        if self.training:
            ret6 = F.interpolate(input=ret6, size=(features[2].shape[2], features[2].shape[3]),
                                 mode='bilinear', align_corners=False)
            ret5 = F.interpolate(input=ret5, size=(features[2].shape[2], features[2].shape[3]),
                                 mode='bilinear', align_corners=False)
            ret4 = F.interpolate(input=ret4, size=(features[2].shape[2], features[2].shape[3]),
                                 mode='bilinear', align_corners=False)
            ret3 = F.interpolate(input=ret3, size=(features[2].shape[2], features[2].shape[3]),
                                 mode='bilinear', align_corners=False)
            ret = F.interpolate(input=ret, size=(features[2].shape[2], features[2].shape[3]),
                                 mode='bilinear', align_corners=False)
            ret = torch.cat((ret6, ret5, ret4, ret3, ret), 1)
        else:
            ret = F.interpolate(input=ret, size=(features[2].shape[2], features[2].shape[3]),
                                 mode='bilinear', align_corners=False)
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2,
                                                                                                ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

class V62Blocker(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

            # Layer 6
            channels6 = 196
            flow_channels6 = 529
            self.process_layer6 = ResBlock(channels6, channels6, layers=4, dilation=2)
            self.upscale_layer6 = ResBlock(channels6, channels6, layers=2, dilation=1)

            # Layer 5
            channels5 = 128
            flow_channels5 = 661
            self.process_layer5 = ResBlock(channels5 + channels6, channels5, layers=4, dilation=2)
            self.upscale_layer5 = ResBlock(channels5, channels5, layers=2, dilation=1)

            # Layer 4
            channels4 = 96
            flow_channels4 = 629
            self.process_layer4 = ResBlock(channels4 + channels5, channels4, layers=4, dilation=2)
            self.upscale_layer4 = ResBlock(channels4, channels4, layers=2, dilation=1)

            # Layer 3
            channels3 = 64
            flow_channels3 = 597
            self.process_layer3 =  ResBlock(channels3 + channels4, channels3, layers=4, dilation=2)


            self.full_process = nn.Sequential(
                ResBlock(channels6 + channels5 + channels4 + channels3, channels3 * 4, layers=3, dilation=2),
                ResBlock(channels3 * 4, channels3 * 2, layers=3, dilation=2),
                ResBlock(channels3 * 2, channels3, layers=3, dilation=2)
            )

            self.output_layer = nn.Conv2d(channels6 + channels5 + channels4 + channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2, flow2, flow_features):
        features = features1

        output = self.process_layer6(features[5])
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)
        result6 = output

        output = self.process_layer5(torch.cat((output, features[4]), 1))
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)
        result5 = output

        output = self.process_layer4(torch.cat((output, features[3]), 1))
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)
        result4 = output

        output = self.process_layer3(torch.cat((output, features[2]), 1))

        output_size = (output.shape[2], output.shape[3])
        output = torch.cat((F.interpolate(input=result6, size=output_size, mode='bilinear', align_corners=False),
                            F.interpolate(input=result5, size=output_size, mode='bilinear', align_corners=False),
                            F.interpolate(input=result4, size=output_size, mode='bilinear', align_corners=False),
                            output), 1)
        output = self.full_process(output)

        output = torch.cat((F.interpolate(input=result6, size=output_size, mode='bilinear', align_corners=False),
                            F.interpolate(input=result5, size=output_size, mode='bilinear', align_corners=False),
                            F.interpolate(input=result4, size=output_size, mode='bilinear', align_corners=False),
                            output), 1)
        ret = self.output_layer(output)

        ret = F.interpolate(input=ret, size=(features[2].shape[2], features[2].shape[3]),
                                 mode='bilinear', align_corners=False)
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2,
                                                                                                ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class V601Blocker(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

            # Layer 6
            channels6 = 196
            flow_channels6 = 529
            self.process_layer6 = nn.Sequential(
                ResBlock(channels6, channels6, layers=2, dilation=2),
                ResBlock(channels6, channels6, layers=2, dilation=2)
            )
            self.upscale_layer6 = ResBlock(channels6, channels6, layers=2, dilation=1)
            self.output_layer6 = nn.Conv2d(channels6, 1, kernel_size=1)

            # Layer 5
            channels5 = 128
            flow_channels5 = 661
            self.process_layer5 = nn.Sequential(
                ResBlock(channels5 + channels6, channels5, layers=2, dilation=2),
                ResBlock(channels5, channels5, layers=2, dilation=2)
            )
            self.upscale_layer5 = ResBlock(channels5, channels5, layers=2, dilation=1)
            self.output_layer5 = nn.Conv2d(channels5, 1, kernel_size=1)

            # Layer 4
            channels4 = 96
            flow_channels4 = 629
            self.process_layer4 = nn.Sequential(
                ResBlock(channels4 + channels5, channels4, layers=2, dilation=2),
                ResBlock(channels4, channels4, layers=2, dilation=2)
            )
            self.upscale_layer4 = ResBlock(channels4, channels4, layers=2, dilation=1)
            self.output_layer4 = nn.Conv2d(channels4, 1, kernel_size=1)

            # Layer 3
            channels3 = 64
            flow_channels3 = 597
            self.process_layer3 =  nn.Sequential(
                ResBlock(channels3 + channels4, channels3, layers=2, dilation=2),
                ResBlock(channels3, channels3, layers=2, dilation=2),
                ResBlock(channels3, channels3, layers=2, dilation=2),
                ResBlock(channels3, channels3, layers=2, dilation=2),
                ResBlock(channels3, channels3, layers=2, dilation=2)
            )

            self.output_layer = nn.Conv2d(channels3, 1, kernel_size=1)

    def cc_forward(self, features1, features2, flow2, flow_features):
        features = features1

        output = self.process_layer6(features[5])
        output = F.interpolate(input=output, size=(features[4].shape[2], features[4].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer6(output)
        ret6 = self.output_layer6(output)

        output = self.process_layer5(torch.cat((output, features[4]), 1))
        output = F.interpolate(input=output, size=(features[3].shape[2], features[3].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer5(output)
        ret5= self.output_layer5(output)

        output = self.process_layer4(torch.cat((output, features[3]), 1))
        output = F.interpolate(input=output, size=(features[2].shape[2], features[2].shape[3]),
                               mode='bilinear', align_corners=False)
        output = self.upscale_layer4(output)
        ret4 = self.output_layer4(output)

        output = self.process_layer3(torch.cat((output, features[2]), 1))
        ret3 = self.output_layer(output)

        ret6 = F.interpolate(input=ret6,
                             size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret5 = F.interpolate(input=ret5,
                             size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret4 = F.interpolate(input=ret4,
                             size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)
        ret3 = F.interpolate(input=ret3,
                             size=(features[2].shape[2], features[2].shape[3]),
                             mode='bilinear', align_corners=False)

        if self.training:
            ret = torch.cat((ret6, ret5, ret4, ret3), 1)
        else:
            ret = ret3
        return ret

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2,
                                                                                                ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density