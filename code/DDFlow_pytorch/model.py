import torch
import sys
import math

from .model_utils import backwarp

# https://github.com/sniklaus/pytorch-pwc

try:
    from .correlation import correlation # the custom cost volume layer
except:
    sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python

class Extractor(torch.nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()

        self.netOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netThr = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netFou = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netSix = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

    # end

    def forward(self, tenInput):
        tenOne = self.netOne(tenInput)
        tenTwo = self.netTwo(tenOne)
        tenThr = self.netThr(tenTwo)
        tenFou = self.netFou(tenThr)
        tenFiv = self.netFiv(tenFou)
        tenSix = self.netSix(tenFiv)

        return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]
# end

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
            torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
        )
    # end

    def forward(self, tenInput):
        return self.netMain(tenInput)
    # end
# end


class PWCNet(torch.nn.Module):
    def __init__(self):
        super(PWCNet, self).__init__()


        self.netExtractor = Extractor()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

    def forward(self, tenFirst, tenSecond):
        tenFirst = self.netExtractor(tenFirst)
        tenSecond = self.netExtractor(tenSecond)

        return self.decode(tenFirst, tenSecond)

    def decode(self, tenFirstFeatures, tenSecondFeatures):
        objEstimate = self.netSix(tenFirstFeatures[-1], tenSecondFeatures[-1], None)
        objEstimate = self.netFiv(tenFirstFeatures[-2], tenSecondFeatures[-2], objEstimate)
        objEstimate = self.netFou(tenFirstFeatures[-3], tenSecondFeatures[-3], objEstimate)
        objEstimate = self.netThr(tenFirstFeatures[-4], tenSecondFeatures[-4], objEstimate)
        objEstimate = self.netTwo(tenFirstFeatures[-5], tenSecondFeatures[-5], objEstimate)

        return objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])

    def full_decode(self, features1, features2, initial_sizes, processed_sizes):
        (int_width, int_height) = initial_sizes
        (int_preprocessed_width, int_preprocessed_height) = processed_sizes
        flow = self.decode(features1, features2)

        # Resize back to original size
        flow = 20.0 * torch.nn.functional.interpolate(input=flow, size=(int_height, int_width),
                                                         mode='bicubic', align_corners=False)

        # Resize weights when rescaling to original size
        flow[:, 0, :, :] *= float(int_width) / float(int_preprocessed_width)
        flow[:, 1, :, :] *= float(int_height) / float(int_preprocessed_height)
        return flow

    def full_forward(self, frame1, frame2, ret_features=False, ret_bw=False):
        int_width = frame1.shape[3]
        int_height = frame1.shape[2]
        int_preprocessed_width = int(math.ceil(math.ceil(int_width / 64.0) * 64.0))
        int_preprocessed_height = int(math.ceil(math.ceil(int_height / 64.0) * 64.0))

        # Resize to get a size which fits into the network
        frame1 = torch.nn.functional.interpolate(input=frame1,
                                                 size=(int_preprocessed_height, int_preprocessed_width),
                                                 mode='bicubic', align_corners=False)
        frame2 = torch.nn.functional.interpolate(input=frame2,
                                                 size=(int_preprocessed_height, int_preprocessed_width),
                                                 mode='bicubic', align_corners=False)

        # Feed through encoder and decode forward and backward
        features1 = self.netExtractor(frame1)
        features2 = self.netExtractor(frame2)
        flow_fw = self.full_decode(features1, features2, initial_sizes=(int_width, int_height),
                                   processed_sizes=(int_preprocessed_width, int_preprocessed_height))
        ret = [flow_fw]

        if ret_bw:
            flow_bw = self.full_decode(features2, features1, initial_sizes=(int_width, int_height),
                                       processed_sizes=(int_preprocessed_width, int_preprocessed_height))
            ret.append(flow_bw)

        if ret_features:
            ret.append(features1)
            ret.append(features2)

        return tuple(ret)

    # Calculate the flow for both forward and backward between the two frames
    def bidirection_forward(self, frame1, frame2, ret_features=False):
        return self.full_forward(frame1, frame2, ret_features=ret_features, ret_bw=True)

    def single_forward(self, frame1, frame2, ret_features = False):
        return self.full_forward(frame1, frame2, ret_features=ret_features, ret_bw=False)
# end