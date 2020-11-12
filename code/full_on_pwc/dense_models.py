import torch.nn as nn
import torch

from guided_filter.guided_filter import GuidedFilter as GuidedFilter

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

class Decoder(torch.nn.Module):
    def __init__(self, intLevel):
        super(Decoder, self).__init__()

        intPrevious = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][
            intLevel + 1]
        intCurrent = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][
            intLevel + 0]

        if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4,
                                                                   stride=2, padding=1)
        if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32,
                                                                   out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 6: self.fltBackwarp = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

        self.netOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netThr = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netFou = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netSix = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1,
                            padding=1)
        )

    # end

    def forward(self, tenFirst, tenSecond, objPrevious):
        tenFlow = None
        tenFeat = None

        if objPrevious is None:
            tenFlow = None
            tenFeat = None

            tenVolume = torch.nn.functional.leaky_relu(
                input=correlation.FunctionCorrelation(tenFirst=tenFirst, tenSecond=tenSecond), negative_slope=0.1,
                inplace=False)

            tenFeat = torch.cat([tenVolume], 1)

        elif objPrevious is not None:
            tenFlow = self.netUpflow(objPrevious['tenFlow'])
            tenFeat = self.netUpfeat(objPrevious['tenFeat'])

            tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=tenFirst,
                                                                                             tenSecond=backwarp(
                                                                                                 tenInput=tenSecond,
                                                                                                 tenFlow=tenFlow * self.fltBackwarp)),
                                                       negative_slope=0.1, inplace=False)

            tenFeat = torch.cat([tenVolume, tenFirst, tenFlow, tenFeat], 1)

        # end

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

class Refiner(torch.nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()

        self.netMain = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3,
                            stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1)
        )
    # end

    def forward(self, tenInput):
        return self.netMain(tenInput)
    # end
# end
# end

class P1Base(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

        #597
        self.flow_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=1, kernel_size=3, stride=1,
                      padding=1)
        )


    def decode(self, features1, features2):
        features = []
        objEstimate = self.netSix(features1[-1], features2[-1], None)
        features.append(objEstimate['tenFeat'])
        objEstimate = self.netFiv(features1[-2], features2[-2], objEstimate)
        features.append(objEstimate['tenFeat'])
        objEstimate = self.netFou(features1[-3], features2[-3], objEstimate)
        features.append(objEstimate['tenFeat'])
        objEstimate = self.netThr(features1[-4], features2[-4], objEstimate)
        features.append(objEstimate['tenFeat'])
        objEstimate = self.netTwo(features1[-5], features2[-5], objEstimate)
        features.append(objEstimate['tenFeat'])

        #self.flow_layer3(features[3])
        return self.flow_layer3(objEstimate['tenFeat']) + self.netRefiner(objEstimate['tenFeat'])

    def cc_forward(self, features1, features2, flow2, flow_features):
        output = self.decode(features1, features2)

        # ret3 = F.interpolate(input=output,
        #                      size=(features1[2].shape[2], features1[2].shape[3]),
        #                      mode='bilinear', align_corners=False)
        # ret = ret3
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density

class Decoder2(torch.nn.Module):
    def __init__(self, intLevel):
        super().__init__()

        intPrevious = [None, None, 32 + 16 + 1, 64 + 16 + 1, 96 + 16 + 1, 128 + 16 + 1, 196, None][
            intLevel + 1]
        intCurrent = [None, None, 32 + 16 + 1, 64 + 16 + 1, 96 + 16 + 1, 128 + 16 + 1, 196, None][
            intLevel + 0]

        if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4,
                                                                   stride=2, padding=1)
        if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32,
                                                                   out_channels=16, kernel_size=4, stride=2, padding=1)

        if intLevel < 4:
            dilation = 2
        else:
            dilation = 1

        self.netOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1,
                            padding=dilation, dilation=dilation),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netThr = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1,
                            padding=dilation, dilation=dilation),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netFou = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1,
                            padding=dilation, dilation=dilation),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1,
                            padding=dilation, dilation=dilation),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netSix = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=1, kernel_size=3, stride=1,
                            padding=1)
        )

    # end

    def forward(self, tenFirst, tenSecond, objPrevious):
        tenFlow = None
        tenFeat = None

        if objPrevious is None:
            tenFlow = None
            tenFeat = None

            tenFeat = torch.cat([tenFirst], 1)

        elif objPrevious is not None:
            tenFlow = self.netUpflow(objPrevious['tenFlow'])
            tenFeat = self.netUpfeat(objPrevious['tenFeat'])

            tenFeat = torch.cat([tenFirst, tenFlow, tenFeat], 1)

        # end

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

class Refiner2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netMain = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64 + 16 + 1 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3,
                            stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1)
        )
    # end

    def forward(self, tenInput):
        return self.netMain(tenInput)
    # end
# end
# end

class P2Base(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        self.netTwo = Decoder2(2)
        self.netThr = Decoder2(3)
        self.netFou = Decoder2(4)
        self.netFiv = Decoder2(5)
        self.netSix = Decoder2(6)

        self.netRefiner = Refiner2()

        #597
        # self.flow_layer3 = nn.Sequential(
        #     nn.Conv2d(in_channels=32 + 16 + 1 + 128 + 128 + 96 + 64 + 32, out_channels=1, kernel_size=3, stride=1,
        #               padding=1)
        # )


    def decode(self, features1, features2):
        features = []
        objEstimate = self.netSix(features1[-1], features2[-1], None)
        features.append(objEstimate['tenFeat'])
        objEstimate = self.netFiv(features1[-2], features2[-2], objEstimate)
        features.append(objEstimate['tenFeat'])
        objEstimate = self.netFou(features1[-3], features2[-3], objEstimate)
        features.append(objEstimate['tenFeat'])
        objEstimate = self.netThr(features1[-4], features2[-4], objEstimate)
        features.append(objEstimate['tenFeat'])
        # objEstimate = self.netTwo(features1[-5], features2[-5], objEstimate)
        # features.append(objEstimate['tenFeat'])

        #self.flow_layer3(features[3])
        return objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])

    def cc_forward(self, features1, features2, flow2, flow_features):
        output = self.decode(features1, features2)

        # ret3 = F.interpolate(input=output,
        #                      size=(features1[2].shape[2], features1[2].shape[3]),
        #                      mode='bilinear', align_corners=False)
        # ret = ret3
        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density
