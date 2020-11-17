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

class Decoder2(torch.nn.Module):
    def __init__(self, intLevel, more_dilation=False):
        super().__init__()

        intPrevious = [None, None, 32 + 16 + 1, 64 + 16 + 1, 96 + 16 + 1, 128 + 16 + 1, 196, None][
            intLevel + 1]
        intCurrent = [None, None, 32 + 16 + 1, 64 + 16 + 1, 96 + 16 + 1, 128 + 16 + 1, 196, None][
            intLevel + 0]

        if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4,
                                                                   stride=2, padding=1)
        if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32,
                                                                   out_channels=16, kernel_size=4, stride=2, padding=1)

        if intLevel < 5:
            multi_dilation = 2
        else:
            multi_dilation = 1

        self.netOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1,
                            padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        if intLevel < 4 and more_dilation:
            self.netThr = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1,
                                padding=multi_dilation, dilation=multi_dilation),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1,
                                padding=multi_dilation, dilation=multi_dilation),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            )

            self.netFou = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=128, kernel_size=3, stride=1,
                                padding=multi_dilation*2, dilation=multi_dilation*2),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1,
                                padding=multi_dilation * 2, dilation=multi_dilation * 2),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
        else:
            self.netThr = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1,
                                padding=multi_dilation, dilation=multi_dilation),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

            self.netFou = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1,
                                padding=multi_dilation * 2, dilation=multi_dilation * 2),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

        self.netFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1,
                            padding=1, dilation=1),
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


    def decode(self, features1, features2):
        features = []
        objEstimate = self.netSix(features1[-1], features2[-1], None)
        #features.append(objEstimate['tenFlow'])
        objEstimate = self.netFiv(features1[-2], features2[-2], objEstimate)
        features.append(objEstimate['tenFlow'])
        objEstimate = self.netFou(features1[-3], features2[-3], objEstimate)
        features.append(objEstimate['tenFlow'])
        objEstimate = self.netThr(features1[-4], features2[-4], objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)

        return output, features

    def cc_forward(self, features1, features2, flow2, flow_features):
        output, features = self.decode(features1, features2)


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

        self.netTwo = Decoder2(2, True)
        self.netThr = Decoder2(3, True)
        self.netFou = Decoder2(4, True)
        self.netFiv = Decoder2(5, True)
        self.netSix = Decoder2(6, True)

        self.netRefiner = Refiner2()


    def decode(self, features1, features2):
        features = []
        objEstimate = self.netSix(features1[-1], features2[-1], None)
        #features.append(objEstimate['tenFlow'])
        objEstimate = self.netFiv(features1[-2], features2[-2], objEstimate)
        features.append(objEstimate['tenFlow'])
        objEstimate = self.netFou(features1[-3], features2[-3], objEstimate)
        features.append(objEstimate['tenFlow'])
        objEstimate = self.netThr(features1[-4], features2[-4], objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)

        return output, features

    def cc_forward(self, features1, features2, flow2, flow_features):
        output, features = self.decode(features1, features2)


        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density



class Decoder3(torch.nn.Module):
    def __init__(self, intLevel, more_dilation=False):
        super().__init__()

        intPrevious = [None, None, 32*2 + 16 + 1, 64*2 + 16 + 1, 96*2 + 16 + 1, 128*2 + 16 + 1, 196*2, None][
            intLevel + 1]
        intCurrent = [None, None, 32*2 + 16 + 1, 64*2 + 16 + 1, 96*2 + 16 + 1, 128*2 + 16 + 1, 196*2, None][
            intLevel + 0]

        if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4,
                                                                   stride=2, padding=1)
        if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32,
                                                                   out_channels=16, kernel_size=4, stride=2, padding=1)

        if intLevel < 5:
            multi_dilation = 2
        else:
            multi_dilation = 1

        self.netOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1,
                            padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        if intLevel < 4 and more_dilation:
            self.netThr = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1,
                                padding=multi_dilation, dilation=multi_dilation),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1,
                                padding=multi_dilation, dilation=multi_dilation),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            )

            self.netFou = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=128, kernel_size=3, stride=1,
                                padding=multi_dilation*2, dilation=multi_dilation*2),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1,
                                padding=multi_dilation * 2, dilation=multi_dilation * 2),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
        else:
            self.netThr = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1,
                                padding=multi_dilation, dilation=multi_dilation),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

            self.netFou = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1,
                                padding=multi_dilation * 2, dilation=multi_dilation * 2),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

        self.netFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1,
                            padding=1, dilation=1),
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

            tenFeat = torch.cat([tenFirst, tenSecond], 1)

        elif objPrevious is not None:
            tenFlow = self.netUpflow(objPrevious['tenFlow'])
            tenFeat = self.netUpfeat(objPrevious['tenFeat'])

            tenFeat = torch.cat([tenFirst, tenSecond, tenFlow, tenFeat], 1)

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

class Refiner3(torch.nn.Module):
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

class P3Base(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        more_dilation = True

        self.netTwo = Decoder2(2, more_dilation)
        self.netThr = Decoder2(3, more_dilation)
        self.netFou = Decoder2(4, more_dilation)
        self.netFiv = Decoder2(5, more_dilation)
        self.netSix = Decoder2(6, more_dilation)

        self.netRefiner = Refiner2()


    def decode(self, features1, features2):
        features = []
        objEstimate = self.netSix(features1[-1], features2[-1], None)
        objEstimate = self.netFiv(features1[-2], features2[-2], objEstimate)
        objEstimate = self.netFou(features1[-3], features2[-3], objEstimate)
        objEstimate = self.netThr(features1[-4], features2[-4], objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)

        return output

    def cc_forward(self, features1, features2, flow2, flow_features):
        output = self.decode(features1, features2)

        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


class Decoder4(torch.nn.Module):
    def __init__(self, intLevel, more_dilation=False):
        super().__init__()

        intPrevious = [None, None, 32*2 + 16 + 1, 64*2 + 16 + 1, 96*2 + 16 + 1, 128*2 + 16 + 1, 196*2, None][
            intLevel + 1]
        intCurrent = [None, None, 32*2 + 16 + 1, 64*2 + 16 + 1, 96*2 + 16 + 1, 128*2 + 16 + 1, 196*2, None][
            intLevel + 0]

        if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4,
                                                                   stride=2, padding=1)
        if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32,
                                                                   out_channels=16, kernel_size=4, stride=2, padding=1)

        if intLevel < 5:
            multi_dilation = 2
        else:
            multi_dilation = 1

        self.netOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1,
                            padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        if intLevel < 4 and more_dilation:
            self.netThr = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1,
                                padding=multi_dilation, dilation=multi_dilation),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1,
                                padding=multi_dilation, dilation=multi_dilation),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            )

            self.netFou = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=128, kernel_size=3, stride=1,
                                padding=multi_dilation*2, dilation=multi_dilation*2),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1,
                                padding=multi_dilation * 2, dilation=multi_dilation * 2),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
        else:
            self.netThr = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1,
                                padding=multi_dilation, dilation=multi_dilation),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

            self.netFou = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1,
                                padding=multi_dilation * 2, dilation=multi_dilation * 2),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

        self.netFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1,
                            padding=1, dilation=1),
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

            tenFeat = torch.cat([tenFirst, tenSecond], 1)

        elif objPrevious is not None:
            tenFlow = self.netUpflow(objPrevious['tenFlow'])
            tenFeat = self.netUpfeat(objPrevious['tenFeat'])

            tenFeat = torch.cat([tenFirst, tenSecond, tenFlow, tenFeat], 1)

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

class Refiner4(torch.nn.Module):
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

class P4Base(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()

        self.fe_net = PWCNet(flow_features=True)

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                         torch.load(path).items()})

        more_dilation = True

        self.netThr = Decoder2(3, more_dilation)
        self.netFou = Decoder2(4, more_dilation)
        self.netFiv = Decoder2(5, more_dilation)
        self.netSix = Decoder2(6, more_dilation)

        self.flowReduceThr = torch.nn.Conv2d(in_channels=597,
                                             out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.flowReduceFou = torch.nn.Conv2d(in_channels=629,
                                             out_channels=96, kernel_size=3, stride=1, padding=1, dilation=1)
        self.flowReduceFiv = torch.nn.Conv2d(in_channels=661,
                                             out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.flowReduceSix = torch.nn.Conv2d(in_channels=529,
                                             out_channels=196, kernel_size=3, stride=1, padding=1, dilation=1)

        self.netRefiner = Refiner2()




    def decode(self, features1, features2):
        objEstimate = self.netSix(features1[-1], self.flowReduceSix(features2[0]), None)
        objEstimate = self.netFiv(features1[-2], self.flowReduceFiv(features2[1]), objEstimate)
        objEstimate = self.netFou(features1[-3], self.flowReduceFou(features2[2]), objEstimate)
        objEstimate = self.netThr(features1[-4], self.flowReduceThr(features2[3]), objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)

        return output

    def cc_forward(self, features1, features2, flow2, flow_features):
        output = self.decode(features1, flow_features)

        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density


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
                torch.nn.Conv2d(in_channels=input_features + layers_features[0] + layers_features[1], out_channels=layers_features[2]*2, kernel_size=3, stride=1,
                                padding=multi_dilation, dilation=multi_dilation),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=layers_features[2]*2, out_channels=layers_features[2], kernel_size=3, stride=1,
                                padding=multi_dilation, dilation=multi_dilation),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            )

            self.netFou = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_features + layers_features[0] + layers_features[1] + layers_features[2], out_channels=layers_features[3]*2, kernel_size=3, stride=1,
                                padding=multi_dilation*2, dilation=multi_dilation*2),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=layers_features[3]*2, out_channels=layers_features[3], kernel_size=3, stride=1,
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

    def decode(self, features1, features2):
        objEstimate = self.netSix(torch.cat([features1[-1], features2[-1]], 1))
        objEstimate = self.netFiv(torch.cat([features1[-2], features2[-2]], 1), objEstimate)
        objEstimate = self.netFou(torch.cat([features1[-3], features2[-3]], 1), objEstimate)
        objEstimate = self.netThr(torch.cat([features1[-4], features2[-4]], 1), objEstimate)
        output = self.netRefiner(objEstimate['tenFeat'])

        if not self.training:
            output = F.relu(output)

        return output

    def cc_forward(self, features1, features2, flow2, flow_features):
        output = self.decode(features1, features2)

        return output

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2, flow_features = self.fe_net.bidirection_forward(frame1, frame2, ret_features=True)
        density = self.cc_forward(features1, features2, flow_fw, flow_features)
        return flow_fw, flow_bw, density