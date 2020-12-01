import torch.nn as nn
import torch

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from DDFlow_pytorch.model import Extractor, PWCNet
import torch.nn.functional as F

from DDFlow_pytorch.model_utils import backwarp
try:
    from .correlation import correlation # the custom cost volume layer
except:
    sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python


import torch
import sys
import math

class DecoderCustom(torch.nn.Module):
    def __init__(self, level, input_features, prev_features=0, bottleneck_features=16,
                 layers_features = [128, 128, 96, 64, 32], more_dilation=False):
        super().__init__()

        self.level = level
        self.input_features = input_features
        self.prev_features = prev_features
        self.bottleneck_features = bottleneck_features
        self.layers_features = layers_features
        self.more_dilation = more_dilation

        if level < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4,
                                                                   stride=2, padding=1)
        if level < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=prev_features,
                                                                   out_channels=bottleneck_features, kernel_size=4, stride=2, padding=1)
        if level < 6:
            input_features = input_features + bottleneck_features + 1

        if level < 5:
            multi_dilation = 2
        else:
            multi_dilation = 1

        self.netOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_features, out_channels=layers_features[0], kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_features + layers_features[0], out_channels=layers_features[1], kernel_size=3, stride=1,
                            padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        if level < 4 and more_dilation:
            self.netThr = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_features + layers_features[0] + layers_features[1], out_channels=layers_features[2], kernel_size=3, stride=1,
                                padding=multi_dilation, dilation=multi_dilation),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=layers_features[2], out_channels=layers_features[2], kernel_size=3, stride=1,
                                padding=multi_dilation, dilation=multi_dilation),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            )

            self.netFou = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_features + layers_features[0] + layers_features[1] + layers_features[2], out_channels=layers_features[3], kernel_size=3, stride=1,
                                padding=multi_dilation * 2, dilation=multi_dilation*2),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=layers_features[3], out_channels=layers_features[3], kernel_size=3, stride=1,
                                padding=multi_dilation * 2, dilation=multi_dilation * 2),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
        else:
            self.netThr = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_features + layers_features[0] + layers_features[1], out_channels=layers_features[2], kernel_size=3, stride=1,
                                padding=multi_dilation, dilation=multi_dilation),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

            self.netFou = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_features + layers_features[0] + layers_features[1] + layers_features[2], out_channels=layers_features[3], kernel_size=3, stride=1,
                                padding=multi_dilation * 2, dilation=multi_dilation * 2),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

        self.netFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_features + layers_features[0] + layers_features[1] + layers_features[2] + layers_features[3], out_channels=layers_features[4], kernel_size=3, stride=1,
                            padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netSix = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_features + layers_features[0] + layers_features[1] + layers_features[2] + layers_features[3] + layers_features[4], out_channels=1, kernel_size=3, stride=1,
                            padding=1)
        )

    # end

    def get_num_output_features(self):
        input_features = self.input_features
        if self.level < 6:
            input_features = input_features + self.bottleneck_features + 1

        return input_features + self.layers_features[0] + self.layers_features[1] + self.layers_features[2] + self.layers_features[3] + self.layers_features[4]

    def forward(self, features, previous=None):
        if previous is None:
            tenFeat = features
        else:
            tenFlow = self.netUpflow(previous['tenFlow'])
            tenFeat = self.netUpfeat(previous['tenFeat'])

            tenFeat = torch.cat([features, tenFlow, tenFeat], 1)

        tenFeat = torch.cat([self.netOne(tenFeat), tenFeat], 1)
        tenFeat = torch.cat([self.netTwo(tenFeat), tenFeat], 1)
        tenFeat = torch.cat([self.netThr(tenFeat), tenFeat], 1)
        tenFeat = torch.cat([self.netFou(tenFeat), tenFeat], 1)
        tenFeat = torch.cat([self.netFiv(tenFeat), tenFeat], 1)

        tenFlow = self.netSix(tenFeat)

        return {
            'tenFlow': tenFlow,
            'tenFeat': tenFeat
        }
    # end
# end


class DecoderCustomSmall(torch.nn.Module):
    def __init__(self, level, input_features, prev_features=0, output_features=-1):
        super().__init__()

        self.level = level
        self.input_features = input_features

        if output_features == -1:
            self.output_features = input_features
        else:
            self.output_features = output_features

        if level < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=prev_features,
                                                                   out_channels=prev_features, kernel_size=4, stride=2, padding=1)
        if level < 6:
            begin_features = input_features + prev_features
        else:
            begin_features = input_features


        self.netOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=begin_features, out_channels=self.output_features, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.output_features, out_channels=self.output_features, kernel_size=3, stride=1,
                            padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
    # end

    def get_num_output_features(self):
        return self.output_features

    def forward(self, features, previous=None):
        if previous is None:
            tenFeat = features
        else:
            tenFeat = self.netUpfeat(previous['tenFeat'])

            tenFeat = torch.cat([features, tenFeat], 1)

        tenFeat = self.netOne(tenFeat)
        tenFeat = self.netTwo(tenFeat)

        return {
            'tenFeat': tenFeat
        }
    # end
# end

class RefinerCustom(torch.nn.Module):
    def __init__(self, input_features, layers_features = [128, 128, 128, 96, 64, 32]):
        super().__init__()

        self.netMain = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_features, out_channels=layers_features[0], kernel_size=3,
                            stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=layers_features[0], out_channels=layers_features[1], kernel_size=3, stride=1, padding=2, dilation=2),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=layers_features[1], out_channels=layers_features[2], kernel_size=3, stride=1, padding=4, dilation=4),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=layers_features[2], out_channels=layers_features[3], kernel_size=3, stride=1, padding=8, dilation=8),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=layers_features[3], out_channels=layers_features[4], kernel_size=3, stride=1, padding=16, dilation=16),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=layers_features[4], out_channels=layers_features[5], kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=layers_features[5], out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1)
        )
    # end

    def forward(self, tenInput):
        return self.netMain(tenInput)
    # end
# end
# end

class P2Small(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        self.netSix = DecoderCustomSmall(6, input_features=196, output_features=196)
        self.netFiv = DecoderCustomSmall(5, input_features=128, output_features=128*2, prev_features=self.netSix.get_num_output_features())
        self.netFou = DecoderCustomSmall(4, input_features=96, output_features=96*2, prev_features=self.netFiv.get_num_output_features())
        self.netThr = DecoderCustomSmall(3, input_features=64, output_features=64*2, prev_features=self.netFou.get_num_output_features())

        self.netRefiner = RefinerCustom(self.netThr.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate = self.netSix(features1[-1], None)
        objEstimate = self.netFiv(features1[-2], objEstimate)
        objEstimate = self.netFou(features1[-3], objEstimate)
        objEstimate = self.netThr(features1[-4], objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)
        
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class P21Small(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        self.netSix = DecoderCustomSmall(6, input_features=196, output_features=196)
        self.netFiv = DecoderCustomSmall(5, input_features=128, output_features=128*2, prev_features=self.netSix.get_num_output_features())
        self.netFou = DecoderCustomSmall(4, input_features=96, output_features=96*2, prev_features=self.netFiv.get_num_output_features())
        self.netThr = DecoderCustomSmall(3, input_features=64, output_features=64*2, prev_features=self.netFou.get_num_output_features())
        self.netTwo = DecoderCustomSmall(3, input_features=32, output_features=64*2, prev_features=self.netThr.get_num_output_features())

        self.netRefiner = RefinerCustom(self.netTwo.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate = self.netSix(features1[-1], None)
        objEstimate = self.netFiv(features1[-2], objEstimate)
        objEstimate = self.netFou(features1[-3], objEstimate)
        objEstimate = self.netThr(features1[-4], objEstimate)
        objEstimate = self.netTwo(features1[-5], objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)
        
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

class P31Small(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        self.netSix = DecoderCustomSmall(6, input_features=196, output_features=196)
        self.netFiv = DecoderCustomSmall(5, input_features=128, output_features=128*2, prev_features=self.netSix.get_num_output_features())
        self.netFou = DecoderCustomSmall(4, input_features=96, output_features=96*2, prev_features=self.netFiv.get_num_output_features())
        self.netThr = DecoderCustomSmall(3, input_features=64, output_features=64*2, prev_features=self.netFou.get_num_output_features())
        self.netTwo = DecoderCustomSmall(3, input_features=32, output_features=64*2, prev_features=self.netThr.get_num_output_features())

        self.netSix2 = DecoderCustomSmall(6, input_features=196, output_features=196)
        self.netFiv2 = DecoderCustomSmall(5, input_features=128, output_features=128*2, prev_features=self.netSix2.get_num_output_features())
        self.netFou2 = DecoderCustomSmall(4, input_features=96, output_features=96*2, prev_features=self.netFiv2.get_num_output_features())
        self.netThr2 = DecoderCustomSmall(3, input_features=64, output_features=64*2, prev_features=self.netFou2.get_num_output_features())
        self.netTwo2 = DecoderCustomSmall(3, input_features=32, output_features=64*2, prev_features=self.netThr2.get_num_output_features())

        self.netRefiner = RefinerCustom(self.netTwo.get_num_output_features()+self.netTwo2.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate = self.netSix(features1[-1], None)
        objEstimate = self.netFiv(features1[-2], objEstimate)
        objEstimate = self.netFou(features1[-3], objEstimate)
        objEstimate = self.netThr(features1[-4], objEstimate)
        objEstimate = self.netTwo(features1[-5], objEstimate)

        objEstimate2 = self.netSix2(features2[-1], None)
        objEstimate2 = self.netFiv2(features2[-2], objEstimate2)
        objEstimate2 = self.netFou2(features2[-3], objEstimate2)
        objEstimate2 = self.netThr2(features2[-4], objEstimate2)
        objEstimate2 = self.netTwo2(features2[-5], objEstimate2)

        output = self.netRefiner(torch.cat([objEstimate['tenFeat'], objEstimate2['tenFeat']], 1))

        if not self.training:
            output = F.relu(output)
        
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class P32Small(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        self.netSix = DecoderCustomSmall(6, input_features=196*2, output_features=196*2)
        self.netFiv = DecoderCustomSmall(5, input_features=128*2, output_features=128*2*2, prev_features=self.netSix.get_num_output_features())
        self.netFou = DecoderCustomSmall(4, input_features=96*2, output_features=96*2*2, prev_features=self.netFiv.get_num_output_features())
        self.netThr = DecoderCustomSmall(3, input_features=64*2, output_features=64*2*2, prev_features=self.netFou.get_num_output_features())
        self.netTwo = DecoderCustomSmall(3, input_features=32*2, output_features=64*2*2, prev_features=self.netThr.get_num_output_features())

        self.netRefiner = RefinerCustom(self.netTwo.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate = self.netSix(torch.cat([features1[-1], features2[-1]], 1), None)
        objEstimate = self.netFiv(torch.cat([features1[-2], features2[-2]], 1), objEstimate)
        objEstimate = self.netFou(torch.cat([features1[-3], features2[-3]], 1), objEstimate)
        objEstimate = self.netThr(torch.cat([features1[-4], features2[-4]], 1), objEstimate)
        objEstimate = self.netTwo(torch.cat([features1[-5], features2[-5]], 1), objEstimate)

        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)
        
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

class P33Small(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        self.netSix = DecoderCustomSmall(6, input_features=196*2, output_features=196*2)
        self.netFiv = DecoderCustomSmall(5, input_features=128*2, output_features=128*2*2, prev_features=self.netSix.get_num_output_features())
        self.netFou = DecoderCustomSmall(4, input_features=96*2, output_features=96*2*2, prev_features=self.netFiv.get_num_output_features())
        self.netThr = DecoderCustomSmall(3, input_features=64*2, output_features=64*2*2, prev_features=self.netFou.get_num_output_features())
        self.netTwo = DecoderCustomSmall(3, input_features=32*2, output_features=64*2*2, prev_features=self.netThr.get_num_output_features())

        self.netSix2 = DecoderCustomSmall(6, input_features=196, output_features=196)
        self.netFiv2 = DecoderCustomSmall(5, input_features=128, output_features=128, prev_features=self.netSix2.get_num_output_features())
        self.netFou2 = DecoderCustomSmall(4, input_features=96, output_features=96, prev_features=self.netFiv2.get_num_output_features())
        self.netThr2 = DecoderCustomSmall(3, input_features=64, output_features=64, prev_features=self.netFou2.get_num_output_features())
        self.netTwo2 = DecoderCustomSmall(3, input_features=32, output_features=32, prev_features=self.netThr2.get_num_output_features())

        self.netRefiner = RefinerCustom(self.netTwo.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate2 = self.netSix2(features2[-1], None)
        objEstimate = self.netSix(torch.cat([features1[-1], objEstimate2['tenFeat']], 1), None)

        objEstimate2 = self.netFiv2(features2[-2], objEstimate2)
        objEstimate = self.netFiv(torch.cat([features1[-2], objEstimate2['tenFeat']], 1), objEstimate)

        objEstimate2 = self.netFou2(features2[-3], objEstimate2)
        objEstimate = self.netFou(torch.cat([features1[-3], objEstimate2['tenFeat']], 1), objEstimate)

        objEstimate2 = self.netThr2(features2[-4], objEstimate2)
        objEstimate = self.netThr(torch.cat([features1[-4], objEstimate2['tenFeat']], 1), objEstimate)

        objEstimate2 = self.netTwo2(features2[-5], objEstimate2)
        objEstimate = self.netTwo(torch.cat([features1[-5], objEstimate2['tenFeat']], 1), objEstimate)

        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)
        
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class P41Small(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        self.netSix = DecoderCustomSmall(6, input_features=196, output_features=196)
        self.netFiv = DecoderCustomSmall(5, input_features=128, output_features=128*2, prev_features=self.netSix.get_num_output_features())
        self.netFou = DecoderCustomSmall(4, input_features=96, output_features=96*2, prev_features=self.netFiv.get_num_output_features())
        self.netThr = DecoderCustomSmall(3, input_features=64, output_features=64*2, prev_features=self.netFou.get_num_output_features())
        self.netTwo = DecoderCustomSmall(3, input_features=32, output_features=64*2, prev_features=self.netThr.get_num_output_features())

        self.netSix2 = DecoderCustomSmall(6, input_features=196, output_features=196)
        self.netFiv2 = DecoderCustomSmall(5, input_features=128, output_features=128*2, prev_features=self.netSix2.get_num_output_features())
        self.netFou2 = DecoderCustomSmall(4, input_features=96, output_features=96*2, prev_features=self.netFiv2.get_num_output_features())
        self.netThr2 = DecoderCustomSmall(3, input_features=64, output_features=64*2, prev_features=self.netFou2.get_num_output_features())
        self.netTwo2 = DecoderCustomSmall(3, input_features=32, output_features=64*2, prev_features=self.netThr2.get_num_output_features())

        self.flowReduceSix = torch.nn.Conv2d(in_channels=529, out_channels=196, kernel_size=3, padding=1)
        self.flowReduceFiv = torch.nn.Conv2d(in_channels=661, out_channels=128, kernel_size=3, padding=1)
        self.flowReduceFou = torch.nn.Conv2d(in_channels=629, out_channels=96, kernel_size=3, padding=1)
        self.flowReduceThr = torch.nn.Conv2d(in_channels=597, out_channels=64, kernel_size=3, padding=1)
        self.flowReduceTwo = torch.nn.Conv2d(in_channels=565, out_channels=32, kernel_size=3, padding=1)

        self.netRefiner = RefinerCustom(self.netTwo.get_num_output_features()+self.netTwo2.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate = self.netSix(features1[-1], None)
        objEstimate = self.netFiv(features1[-2], objEstimate)
        objEstimate = self.netFou(features1[-3], objEstimate)
        objEstimate = self.netThr(features1[-4], objEstimate)
        objEstimate = self.netTwo(features1[-5], objEstimate)

        objEstimate2 = self.netSix2(self.flowReduceSix(flow_features[0]), None)
        objEstimate2 = self.netFiv2(self.flowReduceFiv(flow_features[1]), objEstimate2)
        objEstimate2 = self.netFou2(self.flowReduceFou(flow_features[2]), objEstimate2)
        objEstimate2 = self.netThr2(self.flowReduceThr(flow_features[3]), objEstimate2)
        objEstimate2 = self.netTwo2(self.flowReduceTwo(flow_features[4]), objEstimate2)

        output = self.netRefiner(torch.cat([objEstimate['tenFeat'], objEstimate2['tenFeat']], 1))

        if not self.training:
            output = F.relu(output)
        
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class P42Small(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        self.netSix = DecoderCustomSmall(6, input_features=196*2, output_features=196*2)
        self.netFiv = DecoderCustomSmall(5, input_features=128*2, output_features=128*2*2, prev_features=self.netSix.get_num_output_features())
        self.netFou = DecoderCustomSmall(4, input_features=96*2, output_features=96*2*2, prev_features=self.netFiv.get_num_output_features())
        self.netThr = DecoderCustomSmall(3, input_features=64*2, output_features=64*2*2, prev_features=self.netFou.get_num_output_features())
        self.netTwo = DecoderCustomSmall(3, input_features=32*2, output_features=64*2*2, prev_features=self.netThr.get_num_output_features())

        self.flowReduceSix = torch.nn.Conv2d(in_channels=529, out_channels=196, kernel_size=3, padding=1)
        self.flowReduceFiv = torch.nn.Conv2d(in_channels=661, out_channels=128, kernel_size=3, padding=1)
        self.flowReduceFou = torch.nn.Conv2d(in_channels=629, out_channels=96, kernel_size=3, padding=1)
        self.flowReduceThr = torch.nn.Conv2d(in_channels=597, out_channels=64, kernel_size=3, padding=1)
        self.flowReduceTwo = torch.nn.Conv2d(in_channels=565, out_channels=32, kernel_size=3, padding=1)

        self.netRefiner = RefinerCustom(self.netTwo.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate = self.netSix(torch.cat([features1[-1], self.flowReduceSix(flow_features[0])], 1), None)
        objEstimate = self.netFiv(torch.cat([features1[-2], self.flowReduceFiv(flow_features[1])], 1), objEstimate)
        objEstimate = self.netFou(torch.cat([features1[-3], self.flowReduceFou(flow_features[2])], 1), objEstimate)
        objEstimate = self.netThr(torch.cat([features1[-4], self.flowReduceThr(flow_features[3])], 1), objEstimate)
        objEstimate = self.netTwo(torch.cat([features1[-5], self.flowReduceTwo(flow_features[4])], 1), objEstimate)

        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)
        
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

class P43Small(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        self.netSix = DecoderCustomSmall(6, input_features=196*2, output_features=196*2)
        self.netFiv = DecoderCustomSmall(5, input_features=128*2, output_features=128*2*2, prev_features=self.netSix.get_num_output_features())
        self.netFou = DecoderCustomSmall(4, input_features=96*2, output_features=96*2*2, prev_features=self.netFiv.get_num_output_features())
        self.netThr = DecoderCustomSmall(3, input_features=64*2, output_features=64*2*2, prev_features=self.netFou.get_num_output_features())
        self.netTwo = DecoderCustomSmall(3, input_features=32*2, output_features=64*2*2, prev_features=self.netThr.get_num_output_features())

        self.netSix2 = DecoderCustomSmall(6, input_features=196, output_features=196)
        self.netFiv2 = DecoderCustomSmall(5, input_features=128, output_features=128, prev_features=self.netSix2.get_num_output_features())
        self.netFou2 = DecoderCustomSmall(4, input_features=96, output_features=96, prev_features=self.netFiv2.get_num_output_features())
        self.netThr2 = DecoderCustomSmall(3, input_features=64, output_features=64, prev_features=self.netFou2.get_num_output_features())
        self.netTwo2 = DecoderCustomSmall(3, input_features=32, output_features=32, prev_features=self.netThr2.get_num_output_features())

        self.flowReduceSix = torch.nn.Conv2d(in_channels=529, out_channels=196, kernel_size=3, padding=1)
        self.flowReduceFiv = torch.nn.Conv2d(in_channels=661, out_channels=128, kernel_size=3, padding=1)
        self.flowReduceFou = torch.nn.Conv2d(in_channels=629, out_channels=96, kernel_size=3, padding=1)
        self.flowReduceThr = torch.nn.Conv2d(in_channels=597, out_channels=64, kernel_size=3, padding=1)
        self.flowReduceTwo = torch.nn.Conv2d(in_channels=565, out_channels=32, kernel_size=3, padding=1)

        self.netRefiner = RefinerCustom(self.netTwo.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate2 = self.netSix2(self.flowReduceSix(flow_features[0]), None)
        objEstimate = self.netSix(torch.cat([features1[-1], objEstimate2['tenFeat']], 1), None)

        objEstimate2 = self.netFiv2(self.flowReduceFiv(flow_features[1]), objEstimate2)
        objEstimate = self.netFiv(torch.cat([features1[-2], objEstimate2['tenFeat']], 1), objEstimate)

        objEstimate2 = self.netFou2(self.flowReduceFou(flow_features[2]), objEstimate2)
        objEstimate = self.netFou(torch.cat([features1[-3], objEstimate2['tenFeat']], 1), objEstimate)

        objEstimate2 = self.netThr2(self.flowReduceThr(flow_features[3]), objEstimate2)
        objEstimate = self.netThr(torch.cat([features1[-4], objEstimate2['tenFeat']], 1), objEstimate)

        objEstimate2 = self.netTwo2(self.flowReduceTwo(flow_features[4]), objEstimate2)
        objEstimate = self.netTwo(torch.cat([features1[-5], objEstimate2['tenFeat']], 1), objEstimate)

        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)
        
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

class P61Small(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        self.netSix = DecoderCustomSmall(6, input_features=196, output_features=196)
        self.netFiv = DecoderCustomSmall(5, input_features=128, output_features=128*2, prev_features=self.netSix.get_num_output_features())
        self.netFou = DecoderCustomSmall(4, input_features=96, output_features=96*2, prev_features=self.netFiv.get_num_output_features())
        self.netThr = DecoderCustomSmall(3, input_features=64, output_features=64*2, prev_features=self.netFou.get_num_output_features())
        self.netTwo = DecoderCustomSmall(3, input_features=32, output_features=64*2, prev_features=self.netThr.get_num_output_features())

        self.netSix2 = DecoderCustomSmall(6, input_features=196, output_features=196)
        self.netFiv2 = DecoderCustomSmall(5, input_features=128, output_features=128*2, prev_features=self.netSix2.get_num_output_features())
        self.netFou2 = DecoderCustomSmall(4, input_features=96, output_features=96*2, prev_features=self.netFiv2.get_num_output_features())
        self.netThr2 = DecoderCustomSmall(3, input_features=64, output_features=64*2, prev_features=self.netFou2.get_num_output_features())
        self.netTwo2 = DecoderCustomSmall(3, input_features=32, output_features=64*2, prev_features=self.netThr2.get_num_output_features())

        self.netRefiner = RefinerCustom(self.netTwo.get_num_output_features()+self.netTwo2.get_num_output_features())

    def cc_forward(self, features1, features2, flow, flow_features):
        objEstimate = self.netSix(features1[-1], None)
        objEstimate = self.netFiv(features1[-2], objEstimate)
        objEstimate = self.netFou(features1[-3], objEstimate)
        objEstimate = self.netThr(features1[-4], objEstimate)
        objEstimate = self.netTwo(features1[-5], objEstimate)

        objEstimate2 = self.netSix2(features2[-1], None)
        objEstimate2 = self.netFiv2(features2[-2], objEstimate2)
        objEstimate2 = self.netFou2(features2[-3], objEstimate2)
        objEstimate2 = self.netThr2(features2[-4], objEstimate2)
        objEstimate2 = self.netTwo2(features2[-5], objEstimate2)

        flow = F.interpolate(input=flow, size=(objEstimate2['tenFeat'].shape[2], objEstimate2['tenFeat'].shape[3]),
                              mode='bicubic', align_corners=False)
        flow[:, 0, :, :] = flow[:, 0, :, :] / flow.shape[2] * objEstimate2['tenFeat'].shape[2]
        flow[:, 1, :, :] = flow[:, 1, :, :] / flow.shape[3] * objEstimate2['tenFeat'].shape[3]
        back = backwarp(tenInput=objEstimate2['tenFeat'], tenFlow=flow)

        output = self.netRefiner(torch.cat([objEstimate['tenFeat'], back], 1))

        if not self.training:
            output = F.relu(output)
        
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class P62Small(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        self.netSix = DecoderCustomSmall(6, input_features=196*2, output_features=196*2)
        self.netFiv = DecoderCustomSmall(5, input_features=128*2, output_features=128*2*2, prev_features=self.netSix.get_num_output_features())
        self.netFou = DecoderCustomSmall(4, input_features=96*2, output_features=96*2*2, prev_features=self.netFiv.get_num_output_features())
        self.netThr = DecoderCustomSmall(3, input_features=64*2, output_features=64*2*2, prev_features=self.netFou.get_num_output_features())
        self.netTwo = DecoderCustomSmall(3, input_features=32*2, output_features=64*2*2, prev_features=self.netThr.get_num_output_features())

        self.netRefiner = RefinerCustom(self.netTwo.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        flow = F.interpolate(input=flow2, size=(features2[-1].shape[2], features2[-1].shape[3]),
                              mode='bicubic', align_corners=False)
        flow[:, 0, :, :] = flow[:, 0, :, :] / flow.shape[2] * features2[-1].shape[2]
        flow[:, 1, :, :] = flow[:, 1, :, :] / flow.shape[3] * features2[-1].shape[3]
        back = backwarp(tenInput=features2[-1], tenFlow=flow)
        objEstimate = self.netSix(torch.cat([features1[-1], back], 1), None)

        flow = F.interpolate(input=flow2, size=(features2[-2].shape[2], features2[-2].shape[3]),
                              mode='bicubic', align_corners=False)
        flow[:, 0, :, :] = flow[:, 0, :, :] / flow.shape[2] * features2[-2].shape[2]
        flow[:, 1, :, :] = flow[:, 1, :, :] / flow.shape[3] * features2[-2].shape[3]
        back = backwarp(tenInput=features2[-2], tenFlow=flow)
        objEstimate = self.netFiv(torch.cat([features1[-2], back], 1), objEstimate)

        flow = F.interpolate(input=flow2, size=(features2[-3].shape[2], features2[-3].shape[3]),
                              mode='bicubic', align_corners=False)
        flow[:, 0, :, :] = flow[:, 0, :, :] / flow.shape[2] * features2[-3].shape[2]
        flow[:, 1, :, :] = flow[:, 1, :, :] / flow.shape[3] * features2[-3].shape[3]
        back = backwarp(tenInput=features2[-3], tenFlow=flow)
        objEstimate = self.netFou(torch.cat([features1[-3], back], 1), objEstimate)

        flow = F.interpolate(input=flow2, size=(features2[-4].shape[2], features2[-4].shape[3]),
                              mode='bicubic', align_corners=False)
        flow[:, 0, :, :] = flow[:, 0, :, :] / flow.shape[2] * features2[-4].shape[2]
        flow[:, 1, :, :] = flow[:, 1, :, :] / flow.shape[3] * features2[-4].shape[3]
        back = backwarp(tenInput=features2[-4], tenFlow=flow)
        objEstimate = self.netThr(torch.cat([features1[-4], back], 1), objEstimate)

        flow = F.interpolate(input=flow2, size=(features2[-5].shape[2], features2[-5].shape[3]),
                              mode='bicubic', align_corners=False)
        flow[:, 0, :, :] = flow[:, 0, :, :] / flow.shape[2] * features2[-5].shape[2]
        flow[:, 1, :, :] = flow[:, 1, :, :] / flow.shape[3] * features2[-5].shape[3]
        back = backwarp(tenInput=features2[-5], tenFlow=flow)
        objEstimate = self.netTwo(torch.cat([features1[-5], back], 1), objEstimate)

        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)
        
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class P622Small(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        self.netSix = DecoderCustomSmall(6, input_features=196*2, output_features=196*2)
        self.netFiv = DecoderCustomSmall(5, input_features=128*2, output_features=128*2*2, prev_features=self.netSix.get_num_output_features())
        self.netFou = DecoderCustomSmall(4, input_features=96*2, output_features=96*2*2, prev_features=self.netFiv.get_num_output_features())
        self.netThr = DecoderCustomSmall(3, input_features=64*2, output_features=64*2*2, prev_features=self.netFou.get_num_output_features())
        self.netTwo = DecoderCustomSmall(3, input_features=32*2, output_features=64*2*2, prev_features=self.netThr.get_num_output_features())

        self.netRefiner = RefinerCustom(self.netTwo.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        flow = F.interpolate(input=flow2, size=(features2[-1].shape[2], features2[-1].shape[3]),
                              mode='bicubic', align_corners=False)
        flow[:, 0, :, :] = flow[:, 0, :, :] / flow.shape[2] * features2[-1].shape[2]
        flow[:, 1, :, :] = flow[:, 1, :, :] / flow.shape[3] * features2[-1].shape[3]
        back = backwarp(tenInput=features2[-1], tenFlow=flow)
        objEstimate = self.netSix(torch.cat([features1[-1], back], 1), None)

        flow = F.interpolate(input=flow2, size=(features2[-2].shape[2], features2[-2].shape[3]),
                              mode='bicubic', align_corners=False)
        flow[:, 0, :, :] = flow[:, 0, :, :] / flow.shape[2] * features2[-2].shape[2]
        flow[:, 1, :, :] = flow[:, 1, :, :] / flow.shape[3] * features2[-2].shape[3]
        back = backwarp(tenInput=features2[-2], tenFlow=flow)
        objEstimate = self.netFiv(torch.cat([features1[-2], back], 1), objEstimate)

        flow = F.interpolate(input=flow2, size=(features2[-3].shape[2], features2[-3].shape[3]),
                              mode='bicubic', align_corners=False)
        flow[:, 0, :, :] = flow[:, 0, :, :] / flow.shape[2] * features2[-3].shape[2]
        flow[:, 1, :, :] = flow[:, 1, :, :] / flow.shape[3] * features2[-3].shape[3]
        back = backwarp(tenInput=features2[-3], tenFlow=flow)
        objEstimate = self.netFou(torch.cat([features1[-3], back], 1), objEstimate)

        flow = F.interpolate(input=flow2, size=(features2[-4].shape[2], features2[-4].shape[3]),
                              mode='bicubic', align_corners=False)
        flow[:, 0, :, :] = flow[:, 0, :, :] / flow.shape[2] * features2[-4].shape[2]
        flow[:, 1, :, :] = flow[:, 1, :, :] / flow.shape[3] * features2[-4].shape[3]
        back = backwarp(tenInput=features2[-4], tenFlow=flow)
        objEstimate = self.netThr(torch.cat([features1[-4], back], 1), objEstimate)

        flow = F.interpolate(input=flow2, size=(features2[-5].shape[2], features2[-5].shape[3]),
                              mode='bicubic', align_corners=False)
        flow[:, 0, :, :] = flow[:, 0, :, :] / flow.shape[2] * features2[-5].shape[2]
        flow[:, 1, :, :] = flow[:, 1, :, :] / flow.shape[3] * features2[-5].shape[3]
        back = backwarp(tenInput=features2[-5], tenFlow=flow)
        objEstimate = self.netTwo(torch.cat([features1[-5], back], 1), objEstimate)

        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)
        
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

class P63Small(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        self.netSix = DecoderCustomSmall(6, input_features=196*2, output_features=196*2)
        self.netFiv = DecoderCustomSmall(5, input_features=128*2, output_features=128*2*2, prev_features=self.netSix.get_num_output_features())
        self.netFou = DecoderCustomSmall(4, input_features=96*2, output_features=96*2*2, prev_features=self.netFiv.get_num_output_features())
        self.netThr = DecoderCustomSmall(3, input_features=64*2, output_features=64*2*2, prev_features=self.netFou.get_num_output_features())
        self.netTwo = DecoderCustomSmall(3, input_features=32*2, output_features=64*2*2, prev_features=self.netThr.get_num_output_features())

        self.netSix2 = DecoderCustomSmall(6, input_features=196, output_features=196)
        self.netFiv2 = DecoderCustomSmall(5, input_features=128, output_features=128, prev_features=self.netSix2.get_num_output_features())
        self.netFou2 = DecoderCustomSmall(4, input_features=96, output_features=96, prev_features=self.netFiv2.get_num_output_features())
        self.netThr2 = DecoderCustomSmall(3, input_features=64, output_features=64, prev_features=self.netFou2.get_num_output_features())
        self.netTwo2 = DecoderCustomSmall(3, input_features=32, output_features=32, prev_features=self.netThr2.get_num_output_features())

        self.netRefiner = RefinerCustom(self.netTwo.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate2 = self.netSix2(features2[-1], None)
        flow = F.interpolate(input=flow2, size=(objEstimate2['tenFeat'].shape[2], objEstimate2['tenFeat'].shape[3]),
                              mode='bicubic', align_corners=False)
        flow[:, 0, :, :] = flow[:, 0, :, :] / flow.shape[2] * objEstimate2['tenFeat'].shape[2]
        flow[:, 1, :, :] = flow[:, 1, :, :] / flow.shape[3] * objEstimate2['tenFeat'].shape[3]
        back = backwarp(tenInput=objEstimate2['tenFeat'], tenFlow=flow)
        objEstimate = self.netSix(torch.cat([features1[-1], back], 1), None)

        objEstimate2 = self.netFiv2(features2[-2], objEstimate2)
        flow = F.interpolate(input=flow2, size=(objEstimate2['tenFeat'].shape[2], objEstimate2['tenFeat'].shape[3]),
                              mode='bicubic', align_corners=False)
        flow[:, 0, :, :] = flow[:, 0, :, :] / flow.shape[2] * objEstimate2['tenFeat'].shape[2]
        flow[:, 1, :, :] = flow[:, 1, :, :] / flow.shape[3] * objEstimate2['tenFeat'].shape[3]
        back = backwarp(tenInput=objEstimate2['tenFeat'], tenFlow=flow)
        objEstimate = self.netFiv(torch.cat([features1[-2], back], 1), objEstimate)

        objEstimate2 = self.netFou2(features2[-3], objEstimate2)
        flow = F.interpolate(input=flow2, size=(objEstimate2['tenFeat'].shape[2], objEstimate2['tenFeat'].shape[3]),
                              mode='bicubic', align_corners=False)
        flow[:, 0, :, :] = flow[:, 0, :, :] / flow.shape[2] * objEstimate2['tenFeat'].shape[2]
        flow[:, 1, :, :] = flow[:, 1, :, :] / flow.shape[3] * objEstimate2['tenFeat'].shape[3]
        back = backwarp(tenInput=objEstimate2['tenFeat'], tenFlow=flow)
        objEstimate = self.netFou(torch.cat([features1[-3], back], 1), objEstimate)

        objEstimate2 = self.netThr2(features2[-4], objEstimate2)
        flow = F.interpolate(input=flow2, size=(objEstimate2['tenFeat'].shape[2], objEstimate2['tenFeat'].shape[3]),
                              mode='bicubic', align_corners=False)
        flow[:, 0, :, :] = flow[:, 0, :, :] / flow.shape[2] * objEstimate2['tenFeat'].shape[2]
        flow[:, 1, :, :] = flow[:, 1, :, :] / flow.shape[3] * objEstimate2['tenFeat'].shape[3]
        back = backwarp(tenInput=objEstimate2['tenFeat'], tenFlow=flow)
        objEstimate = self.netThr(torch.cat([features1[-4], back], 1), objEstimate)

        objEstimate2 = self.netTwo2(features2[-5], objEstimate2)
        flow = F.interpolate(input=flow2, size=(objEstimate2['tenFeat'].shape[2], objEstimate2['tenFeat'].shape[3]),
                              mode='bicubic', align_corners=False)
        flow[:, 0, :, :] = flow[:, 0, :, :] / flow.shape[2] * objEstimate2['tenFeat'].shape[2]
        flow[:, 1, :, :] = flow[:, 1, :, :] / flow.shape[3] * objEstimate2['tenFeat'].shape[3]
        back = backwarp(tenInput=objEstimate2['tenFeat'], tenFlow=flow)
        objEstimate = self.netTwo(torch.cat([features1[-5], back], 1), objEstimate)

        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)
        
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

########################################################
#### REGULAR SIZE, but not performing very well.... ####
#### Or just on par with the smaller sizes...       ####
########################################################
class P2Base(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        more_dilation = False 

        self.netSix = DecoderCustom(6, input_features=196, more_dilation=more_dilation)
        self.netFiv = DecoderCustom(5, input_features=128, prev_features=self.netSix.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netFou = DecoderCustom(4, input_features=96, prev_features=self.netFiv.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netThr = DecoderCustom(3, input_features=64, prev_features=self.netFou.get_num_output_features(),
                                    more_dilation=more_dilation)

        self.netRefiner = RefinerCustom(self.netThr.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate = self.netSix(features1[-1], None)
        objEstimate = self.netFiv(features1[-2], objEstimate)
        objEstimate = self.netFou(features1[-3], objEstimate)
        objEstimate = self.netThr(features1[-4], objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)

        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

class P21Base(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        more_dilation = True 

        self.netSix = DecoderCustom(6, input_features=196, more_dilation=more_dilation)
        self.netFiv = DecoderCustom(5, input_features=128, prev_features=self.netSix.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netFou = DecoderCustom(4, input_features=96, prev_features=self.netFiv.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netThr = DecoderCustom(3, input_features=64, prev_features=self.netFou.get_num_output_features(),
                                    more_dilation=more_dilation)

        self.netRefiner = RefinerCustom(self.netThr.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate = self.netSix(features1[-1], None)
        objEstimate = self.netFiv(features1[-2], objEstimate)
        objEstimate = self.netFou(features1[-3], objEstimate)
        objEstimate = self.netThr(features1[-4], objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)
        
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class P3Base(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        more_dilation = False

        self.netSix = DecoderCustom(6, input_features=196*2, more_dilation=more_dilation)
        self.netFiv = DecoderCustom(5, input_features=128*2, prev_features=self.netSix.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netFou = DecoderCustom(4, input_features=96*2, prev_features=self.netFiv.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netThr = DecoderCustom(3, input_features=64*2, prev_features=self.netFou.get_num_output_features(),
                                    more_dilation=more_dilation)

        self.netRefiner = RefinerCustom(self.netThr.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate = self.netSix(torch.cat([features1[-1], features2[-1]], 1), None)
        objEstimate = self.netFiv(torch.cat([features1[-2], features2[-2]], 1), objEstimate)
        objEstimate = self.netFou(torch.cat([features1[-3], features2[-3]], 1), objEstimate)
        objEstimate = self.netThr(torch.cat([features1[-4], features2[-4]], 1), objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)
        
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class P31Base(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        more_dilation = True

        self.netSix = DecoderCustom(6, input_features=196*2, more_dilation=more_dilation)
        self.netFiv = DecoderCustom(5, input_features=128*2, prev_features=self.netSix.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netFou = DecoderCustom(4, input_features=96*2, prev_features=self.netFiv.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netThr = DecoderCustom(3, input_features=64*2, prev_features=self.netFou.get_num_output_features(),
                                    more_dilation=more_dilation)

        self.netRefiner = RefinerCustom(self.netThr.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate = self.netSix(torch.cat([features1[-1], features2[-1]], 1), None)
        objEstimate = self.netFiv(torch.cat([features1[-2], features2[-2]], 1), objEstimate)
        objEstimate = self.netFou(torch.cat([features1[-3], features2[-3]], 1), objEstimate)
        objEstimate = self.netThr(torch.cat([features1[-4], features2[-4]], 1), objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)

        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

class P32Base(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        more_dilation = True 

        self.netSix = DecoderCustom(6, input_features=196, more_dilation=more_dilation)
        self.netFiv = DecoderCustom(5, input_features=128, prev_features=self.netSix.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netFou = DecoderCustom(4, input_features=96, prev_features=self.netFiv.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netThr = DecoderCustom(3, input_features=64, prev_features=self.netFou.get_num_output_features(),
                                    more_dilation=more_dilation)

        self.netSix2 = DecoderCustom(6, input_features=196, more_dilation=more_dilation)
        self.netFiv2 = DecoderCustom(5, input_features=128, prev_features=self.netSix.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netFou2 = DecoderCustom(4, input_features=96, prev_features=self.netFiv.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netThr2 = DecoderCustom(3, input_features=64, prev_features=self.netFou.get_num_output_features(),
                                    more_dilation=more_dilation)

        self.netRefiner = RefinerCustom(self.netThr.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate = self.netSix(features1[-1], None)
        objEstimate2 = self.netSix2(features2[-1], None)
        objEstimate['tenFeat'] = objEstimate['tenFeat'] + objEstimate2['tenFeat']

        objEstimate = self.netFiv(features1[-2], objEstimate)
        objEstimate2 = self.netFiv2(features2[-2], objEstimate2)
        objEstimate['tenFeat'] = objEstimate['tenFeat'] + objEstimate2['tenFeat']

        objEstimate = self.netFou(features1[-3], objEstimate)
        objEstimate2 = self.netFou2(features2[-3], objEstimate2)
        objEstimate['tenFeat'] = objEstimate['tenFeat'] + objEstimate2['tenFeat']

        objEstimate = self.netThr(features1[-4], objEstimate)
        objEstimate2 = self.netThr2(features2[-4], objEstimate2)
        objEstimate['tenFeat'] = objEstimate['tenFeat'] + objEstimate2['tenFeat']

        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)
        
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class P4Base(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        more_dilation = False
        self.netSix = DecoderCustom(6, input_features=196*2, more_dilation=more_dilation)
        self.netFiv = DecoderCustom(5, input_features=128*2, prev_features=self.netSix.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netFou = DecoderCustom(4, input_features=96*2, prev_features=self.netFiv.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netThr = DecoderCustom(3, input_features=64*2, prev_features=self.netFou.get_num_output_features(),
                                    more_dilation=more_dilation)

        self.flowReduceThr = torch.nn.Conv2d(in_channels=597, out_channels=64, kernel_size=3, padding=1)
        self.flowReduceFou = torch.nn.Conv2d(in_channels=629, out_channels=96, kernel_size=3, padding=1)
        self.flowReduceFiv = torch.nn.Conv2d(in_channels=661, out_channels=128, kernel_size=3, padding=1)
        self.flowReduceSix = torch.nn.Conv2d(in_channels=529, out_channels=196, kernel_size=3, padding=1)

        self.netRefiner = RefinerCustom(self.netThr.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate = self.netSix(torch.cat([features1[-1], self.flowReduceSix(flow_features[0])], 1), None)
        objEstimate = self.netFiv(torch.cat([features1[-2], self.flowReduceFiv(flow_features[1])], 1), objEstimate)
        objEstimate = self.netFou(torch.cat([features1[-3], self.flowReduceFou(flow_features[2])], 1), objEstimate)
        objEstimate = self.netThr(torch.cat([features1[-4], self.flowReduceThr(flow_features[3])], 1), objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)

        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

class P41Base(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        more_dilation = True
        self.netSix = DecoderCustom(6, input_features=196*2, more_dilation=more_dilation)
        self.netFiv = DecoderCustom(5, input_features=128*2, prev_features=self.netSix.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netFou = DecoderCustom(4, input_features=96*2, prev_features=self.netFiv.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netThr = DecoderCustom(3, input_features=64*2, prev_features=self.netFou.get_num_output_features(),
                                    more_dilation=more_dilation)

        self.flowReduceThr = torch.nn.Conv2d(in_channels=597, out_channels=64, kernel_size=3, padding=1)
        self.flowReduceFou = torch.nn.Conv2d(in_channels=629, out_channels=96, kernel_size=3, padding=1)
        self.flowReduceFiv = torch.nn.Conv2d(in_channels=661, out_channels=128, kernel_size=3, padding=1)
        self.flowReduceSix = torch.nn.Conv2d(in_channels=529, out_channels=196, kernel_size=3, padding=1)

        self.netRefiner = RefinerCustom(self.netThr.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate = self.netSix(torch.cat([features1[-1], self.flowReduceSix(flow_features[0])], 1), None)
        objEstimate = self.netFiv(torch.cat([features1[-2], self.flowReduceFiv(flow_features[1])], 1), objEstimate)
        objEstimate = self.netFou(torch.cat([features1[-3], self.flowReduceFou(flow_features[2])], 1), objEstimate)
        objEstimate = self.netThr(torch.cat([features1[-4], self.flowReduceThr(flow_features[3])], 1), objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)

        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

class P5Base(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        more_dilation = False
        self.netSix = DecoderCustom(6, input_features=196*3, more_dilation=more_dilation)
        self.netFiv = DecoderCustom(5, input_features=128*3, prev_features=self.netSix.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netFou = DecoderCustom(4, input_features=96*3, prev_features=self.netFiv.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netThr = DecoderCustom(3, input_features=64*3, prev_features=self.netFou.get_num_output_features(),
                                    more_dilation=more_dilation)

        self.flowReduceThr = torch.nn.Conv2d(in_channels=597, out_channels=64, kernel_size=3, padding=1)
        self.flowReduceFou = torch.nn.Conv2d(in_channels=629, out_channels=96, kernel_size=3, padding=1)
        self.flowReduceFiv = torch.nn.Conv2d(in_channels=661, out_channels=128, kernel_size=3, padding=1)
        self.flowReduceSix = torch.nn.Conv2d(in_channels=529, out_channels=196, kernel_size=3, padding=1)

        self.netRefiner = RefinerCustom(self.netThr.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate = self.netSix(torch.cat([features1[-1], features2[-1], self.flowReduceSix(flow_features[0])], 1), None)
        objEstimate = self.netFiv(torch.cat([features1[-2], features2[-2], self.flowReduceFiv(flow_features[1])], 1), objEstimate)
        objEstimate = self.netFou(torch.cat([features1[-3], features2[-3], self.flowReduceFou(flow_features[2])], 1), objEstimate)
        objEstimate = self.netThr(torch.cat([features1[-4], features2[-4], self.flowReduceThr(flow_features[3])], 1), objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)

        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class P51Base(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        more_dilation = False
        self.netSix = DecoderCustom(6, input_features=196*3, more_dilation=more_dilation)
        self.netFiv = DecoderCustom(5, input_features=128*3, prev_features=self.netSix.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netFou = DecoderCustom(4, input_features=96*3, prev_features=self.netFiv.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netThr = DecoderCustom(3, input_features=64*3, prev_features=self.netFou.get_num_output_features(),
                                    more_dilation=more_dilation)

        self.flowReduceThr = torch.nn.Conv2d(in_channels=597, out_channels=64, kernel_size=3, padding=1)
        self.flowReduceFou = torch.nn.Conv2d(in_channels=629, out_channels=96, kernel_size=3, padding=1)
        self.flowReduceFiv = torch.nn.Conv2d(in_channels=661, out_channels=128, kernel_size=3, padding=1)
        self.flowReduceSix = torch.nn.Conv2d(in_channels=529, out_channels=196, kernel_size=3, padding=1)

        self.netRefiner = RefinerCustom(self.netThr.get_num_output_features())

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate = self.netSix(torch.cat([features1[-1], features2[-1], self.flowReduceSix(flow_features[0])], 1), None)
        objEstimate = self.netFiv(torch.cat([features1[-2], features2[-2], self.flowReduceFiv(flow_features[1])], 1), objEstimate)
        objEstimate = self.netFou(torch.cat([features1[-3], features2[-3], self.flowReduceFou(flow_features[2])], 1), objEstimate)
        objEstimate = self.netThr(torch.cat([features1[-4], features2[-4], self.flowReduceThr(flow_features[3])], 1), objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)

        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class P52Base(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        fm = 2
        bf = int(fm*16) # Bottleneck features

        layers_features = [int(fm*128), int(fm*128), int(fm*96), int(fm*64), int(fm*32)]
        self.netSix = DecoderCustom(6, input_features=196 * 3, layers_features=layers_features,
                                    bottleneck_features=bf, more_dilation=more_dilation)
        self.netFiv = DecoderCustom(5, input_features=128 * 3, prev_features=self.netSix.get_num_output_features(),
                                    layers_features=layers_features, bottleneck_features=bf, more_dilation=more_dilation)
        self.netFou = DecoderCustom(4, input_features=96 * 3, prev_features=self.netFiv.get_num_output_features(),
                                    layers_features=layers_features, bottleneck_features=bf, more_dilation=more_dilation)
        self.netThr = DecoderCustom(3, input_features=64 * 3, prev_features=self.netFou.get_num_output_features(),
                                    layers_features=layers_features, bottleneck_features=bf, more_dilation=more_dilation)
        
        layers_features = [int(fm*128), int(fm*128), int(fm*128), int(fm*96), int(fm*64), int(fm*32)]
        self.netRefiner = RefinerCustom(input_features=self.netThr.get_num_output_features(), layers_features=layers_features)
    
        self.flowReduceSix = torch.nn.Conv2d(in_channels=529, out_channels=196, kernel_size=3, padding=1)
        self.flowReduceFiv = torch.nn.Conv2d(in_channels=661, out_channels=128, kernel_size=3, padding=1)
        self.flowReduceFou = torch.nn.Conv2d(in_channels=629, out_channels=96, kernel_size=3, padding=1)
        self.flowReduceThr = torch.nn.Conv2d(in_channels=597, out_channels=64, kernel_size=3, padding=1)

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate = self.netSix(torch.cat([features1[-1], features2[-1], self.flowReduceSix(flow_features[0])], 1), None)
        objEstimate = self.netFiv(torch.cat([features1[-2], features2[-2], self.flowReduceFiv(flow_features[1])], 1), objEstimate)
        objEstimate = self.netFou(torch.cat([features1[-3], features2[-3], self.flowReduceFou(flow_features[2])], 1), objEstimate)
        objEstimate = self.netThr(torch.cat([features1[-4], features2[-4], self.flowReduceThr(flow_features[3])], 1), objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)

        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class P61Base(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        # Variable that adds the second density as loss as well
        self.needs_second_density = True

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        more_dilation = True

        self.netSix = DecoderCustom(6, input_features=196*2, more_dilation=more_dilation)
        self.netFiv = DecoderCustom(5, input_features=128*2, prev_features=self.netSix.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netFou = DecoderCustom(4, input_features=96*2, prev_features=self.netFiv.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netThr = DecoderCustom(3, input_features=64*2, prev_features=self.netFou.get_num_output_features(),
                                    more_dilation=more_dilation)

        self.netRefiner = RefinerCustom(self.netThr.get_num_output_features())

    def cc_forward(self, features1, features2, flow, flow_features):
        flow6 = F.interpolate(input=flow, size=(features2[-1].shape[2], features2[-1].shape[3]),
                              mode='bicubic', align_corners=False)
        flow6[:, 0, :, :] = flow6[:, 0, :, :] / flow.shape[2] * features2[-1].shape[2]
        flow6[:, 1, :, :] = flow6[:, 1, :, :] / flow.shape[3] * features2[-1].shape[3]
        back6 = backwarp(tenInput=features2[-1], tenFlow=flow6)
        objEstimate = self.netSix(torch.cat([features1[-1], back6], 1), None)

        flow5 = F.interpolate(input=flow, size=(features2[-2].shape[2], features2[-2].shape[3]),
                              mode='bicubic', align_corners=False)
        flow5[:, 0, :, :] = flow5[:, 0, :, :] / flow.shape[2] * features2[-2].shape[2]
        flow5[:, 1, :, :] = flow5[:, 1, :, :] / flow.shape[3] * features2[-2].shape[3]
        back5 = backwarp(tenInput=features2[-2], tenFlow=flow5)
        objEstimate = self.netFiv(torch.cat([features1[-2], back5], 1), objEstimate)

        flow4 = F.interpolate(input=flow, size=(features2[-3].shape[2], features2[-3].shape[3]),
                              mode='bicubic', align_corners=False)
        flow4[:, 0, :, :] = flow4[:, 0, :, :] / flow.shape[2] * features2[-3].shape[2]
        flow4[:, 1, :, :] = flow4[:, 1, :, :] / flow.shape[3] * features2[-3].shape[3]
        back4 = backwarp(tenInput=features2[-3], tenFlow=flow4)
        objEstimate = self.netFou(torch.cat([features1[-3], back4], 1), objEstimate)

        flow3 = F.interpolate(input=flow, size=(features2[-4].shape[2], features2[-4].shape[3]),
                              mode='bicubic', align_corners=False)
        flow3[:, 0, :, :] = flow3[:, 0, :, :] / flow.shape[2] * features2[-4].shape[2]
        flow3[:, 1, :, :] = flow3[:, 1, :, :] / flow.shape[3] * features2[-4].shape[3]
        back3 = backwarp(tenInput=features2[-4], tenFlow=flow3)
        objEstimate = self.netThr(torch.cat([features1[-4], back3], 1), objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)

        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

class P62Base(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        more_dilation = True 

        self.netSix = DecoderCustom(6, input_features=196, more_dilation=more_dilation)
        self.netFiv = DecoderCustom(5, input_features=128, prev_features=self.netSix.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netFou = DecoderCustom(4, input_features=96, prev_features=self.netFiv.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netThr = DecoderCustom(3, input_features=64, prev_features=self.netFou.get_num_output_features(),
                                    more_dilation=more_dilation)

        self.netSix2 = DecoderCustom(6, input_features=196, more_dilation=more_dilation)
        self.netFiv2 = DecoderCustom(5, input_features=128, prev_features=self.netSix.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netFou2 = DecoderCustom(4, input_features=96, prev_features=self.netFiv.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netThr2 = DecoderCustom(3, input_features=64, prev_features=self.netFou.get_num_output_features(),
                                    more_dilation=more_dilation)

        self.netRefiner = RefinerCustom(self.netThr.get_num_output_features())

    def cc_forward(self, features1, features2, flow, flow_features):
        objEstimate = self.netSix(features1[-1], None)
        objEstimate2 = self.netSix2(features2[-1], None)
        flow6 = F.interpolate(input=flow, size=(features2[-1].shape[2], features2[-1].shape[3]),
                              mode='bicubic', align_corners=False)
        flow6[:, 0, :, :] = flow6[:, 0, :, :] / flow.shape[2] * features2[-1].shape[2]
        flow6[:, 1, :, :] = flow6[:, 1, :, :] / flow.shape[3] * features2[-1].shape[3]
        back6 = backwarp(tenInput=objEstimate2['tenFeat'], tenFlow=flow6)
        objEstimate['tenFeat'] = objEstimate['tenFeat'] + back6

        objEstimate = self.netFiv(features1[-2], objEstimate)
        objEstimate2 = self.netFiv2(features2[-2], objEstimate2)
        flow5 = F.interpolate(input=flow, size=(features2[-2].shape[2], features2[-2].shape[3]),
                              mode='bicubic', align_corners=False)
        flow5[:, 0, :, :] = flow5[:, 0, :, :] / flow.shape[2] * features2[-2].shape[2]
        flow5[:, 1, :, :] = flow5[:, 1, :, :] / flow.shape[3] * features2[-2].shape[3]
        back5 = backwarp(tenInput=objEstimate2['tenFeat'], tenFlow=flow5)
        objEstimate['tenFeat'] = objEstimate['tenFeat'] + back5

        objEstimate = self.netFou(features1[-3], objEstimate)
        objEstimate2 = self.netFou2(features2[-3], objEstimate2)
        flow4 = F.interpolate(input=flow, size=(features2[-3].shape[2], features2[-3].shape[3]),
                              mode='bicubic', align_corners=False)
        flow4[:, 0, :, :] = flow4[:, 0, :, :] / flow.shape[2] * features2[-3].shape[2]
        flow4[:, 1, :, :] = flow4[:, 1, :, :] / flow.shape[3] * features2[-3].shape[3]
        back4 = backwarp(tenInput=objEstimate2['tenFeat'], tenFlow=flow4)
        objEstimate['tenFeat'] = objEstimate['tenFeat'] + back4

        objEstimate = self.netThr(features1[-4], objEstimate)
        objEstimate2 = self.netThr2(features2[-4], objEstimate2)
        flow3 = F.interpolate(input=flow, size=(features2[-4].shape[2], features2[-4].shape[3]),
                              mode='bicubic', align_corners=False)
        flow3[:, 0, :, :] = flow3[:, 0, :, :] / flow.shape[2] * features2[-4].shape[2]
        flow3[:, 1, :, :] = flow3[:, 1, :, :] / flow.shape[3] * features2[-4].shape[3]
        back3 = backwarp(tenInput=objEstimate2['tenFeat'], tenFlow=flow3)
        objEstimate['tenFeat'] = objEstimate['tenFeat'] + back3

        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)
        
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

class P622Base(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        more_dilation = True 

        self.netSix = DecoderCustom(6, input_features=196, more_dilation=more_dilation)
        self.netFiv = DecoderCustom(5, input_features=128, prev_features=self.netSix.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netFou = DecoderCustom(4, input_features=96, prev_features=self.netFiv.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netThr = DecoderCustom(3, input_features=64, prev_features=self.netFou.get_num_output_features(),
                                    more_dilation=more_dilation)

        self.netSix2 = DecoderCustom(6, input_features=196, more_dilation=more_dilation)
        self.netFiv2 = DecoderCustom(5, input_features=128, prev_features=self.netSix.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netFou2 = DecoderCustom(4, input_features=96, prev_features=self.netFiv.get_num_output_features(),
                                    more_dilation=more_dilation)
        self.netThr2 = DecoderCustom(3, input_features=64, prev_features=self.netFou.get_num_output_features(),
                                    more_dilation=more_dilation)

        self.netRefiner = RefinerCustom(self.netThr.get_num_output_features())
        self.netRefiner2 = RefinerCustom(self.netThr.get_num_output_features())

    def cc_forward(self, features1, features2, flow, flow_features):
        objEstimate = self.netSix(features1[-1], None)
        objEstimate2 = self.netSix2(features2[-1], None)
        flow6 = F.interpolate(input=flow, size=(features2[-1].shape[2], features2[-1].shape[3]),
                              mode='bicubic', align_corners=False)
        flow6[:, 0, :, :] = flow6[:, 0, :, :] / flow.shape[2] * features2[-1].shape[2]
        flow6[:, 1, :, :] = flow6[:, 1, :, :] / flow.shape[3] * features2[-1].shape[3]
        back6 = backwarp(tenInput=objEstimate2['tenFeat'], tenFlow=flow6)
        objEstimate['tenFeat'] = objEstimate['tenFeat'] + back6

        objEstimate = self.netFiv(features1[-2], objEstimate)
        objEstimate2 = self.netFiv2(features2[-2], objEstimate2)
        flow5 = F.interpolate(input=flow, size=(features2[-2].shape[2], features2[-2].shape[3]),
                              mode='bicubic', align_corners=False)
        flow5[:, 0, :, :] = flow5[:, 0, :, :] / flow.shape[2] * features2[-2].shape[2]
        flow5[:, 1, :, :] = flow5[:, 1, :, :] / flow.shape[3] * features2[-2].shape[3]
        back5 = backwarp(tenInput=objEstimate2['tenFeat'], tenFlow=flow5)
        objEstimate['tenFeat'] = objEstimate['tenFeat'] + back5

        objEstimate = self.netFou(features1[-3], objEstimate)
        objEstimate2 = self.netFou2(features2[-3], objEstimate2)
        flow4 = F.interpolate(input=flow, size=(features2[-3].shape[2], features2[-3].shape[3]),
                              mode='bicubic', align_corners=False)
        flow4[:, 0, :, :] = flow4[:, 0, :, :] / flow.shape[2] * features2[-3].shape[2]
        flow4[:, 1, :, :] = flow4[:, 1, :, :] / flow.shape[3] * features2[-3].shape[3]
        back4 = backwarp(tenInput=objEstimate2['tenFeat'], tenFlow=flow4)
        objEstimate['tenFeat'] = objEstimate['tenFeat'] + back4

        objEstimate = self.netThr(features1[-4], objEstimate)
        objEstimate2 = self.netThr2(features2[-4], objEstimate2)
        flow3 = F.interpolate(input=flow, size=(features2[-4].shape[2], features2[-4].shape[3]),
                              mode='bicubic', align_corners=False)
        flow3[:, 0, :, :] = flow3[:, 0, :, :] / flow.shape[2] * features2[-4].shape[2]
        flow3[:, 1, :, :] = flow3[:, 1, :, :] / flow.shape[3] * features2[-4].shape[3]
        back3 = backwarp(tenInput=objEstimate2['tenFeat'], tenFlow=flow3)
        objEstimate['tenFeat'] = objEstimate['tenFeat'] + back3

        output = self.netRefiner(objEstimate['tenFeat'])

        if self.training:
            output = torch.cat([output, self.netRefiner2(objEstimate2['tenFeat'])], 1)
        else:
            output = F.relu(output)
        
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

class PCustom(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        more_dilation = True
        fm = 1.5
        bf = 32 # int(fm*16) # Bottleneck features

        layers_features = [int(fm*128), int(fm*128), int(fm*96), int(fm*64), int(fm*32)]
        self.netSix = DecoderCustom(6, input_features=196 * 2, layers_features=layers_features,
                                    bottleneck_features=bf, more_dilation=more_dilation)
        self.netFiv = DecoderCustom(5, input_features=128 * 2, prev_features=self.netSix.get_num_output_features(),
                                    layers_features=layers_features, bottleneck_features=bf, more_dilation=more_dilation)
        self.netFou = DecoderCustom(4, input_features=96 * 2, prev_features=self.netFiv.get_num_output_features(),
                                    layers_features=layers_features, bottleneck_features=bf, more_dilation=more_dilation)
        self.netThr = DecoderCustom(3, input_features=64*2, prev_features=self.netFou.get_num_output_features(),
                                    layers_features=layers_features, bottleneck_features=bf, more_dilation=more_dilation)
        
        layers_features = [int(fm*128), int(fm*128), int(fm*128), int(fm*96), int(fm*64), int(fm*32)]
        self.netRefiner = RefinerCustom(input_features=self.netThr.get_num_output_features(), layers_features=layers_features)

    def cc_forward(self, features1, features2, flow2, flow_features):
        objEstimate = self.netSix(torch.cat([features1[-1], features2[-1]], 1))
        objEstimate = self.netFiv(torch.cat([features1[-2], features2[-2]], 1), objEstimate)
        objEstimate = self.netFou(torch.cat([features1[-3], features2[-3]], 1), objEstimate)
        objEstimate = self.netThr(torch.cat([features1[-4], features2[-4]], 1), objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)

        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density