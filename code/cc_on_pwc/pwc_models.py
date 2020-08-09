import torch.nn as nn
import torch

from model_utils import ConvBlock, DeconvBlock

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from DDFlow_pytorch.model import Extractor
from guided_filter.guided_filter import GuidedFilter as GuidedFilter


class PWCC_V1(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        channels = 128

        self.extractor = Extractor()
        if pretrained:
            self.extractor.load_state_dict(torch.load('chairs_extractor.pt'))

        self.output_layer = nn.Conv2d(channels, 1, kernel_size=1)

        self.process_layer3 = nn.Sequential(
            ConvBlock(64, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )


    def forward(self, input):
        features = self.extractor(input)
        input = features[2]

        output = self.process_layer3(input)
        return self.output_layer(output)

class PWCC_V2_small(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        channels = 128

        self.extractor = Extractor()
        if pretrained:
            self.extractor.load_state_dict(torch.load('chairs_extractor.pt'))

        self.output_layer = nn.Conv2d(16, 1, kernel_size=1)

        self.process_layer3 = nn.Sequential(
            ConvBlock(64, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer3 = nn.Sequential(
            DeconvBlock(channels, 32, kernel=4, stride=2),
            ConvBlock(32, 32, kernel=3, stride=1, dilation=1),
            ConvBlock(32, 32, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer2 = nn.Sequential(
            DeconvBlock(32, 16, kernel=4, stride=2),
            ConvBlock(16, 16, kernel=3, stride=1, dilation=1),
            ConvBlock(16, 16, kernel=3, stride=1, dilation=1),
        )


    def forward(self, input):
        features = self.extractor(input)
        input = features[2]

        output = self.process_layer3(input)
        output = self.upscale_layer3(output)
        output = self.upscale_layer2(output)
        return self.output_layer(output)

class PWCC_V1_deep(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        channels = 128

        self.extractor = Extractor()
        if pretrained:
            self.extractor.load_state_dict(torch.load('chairs_extractor.pt'))

        self.output_layer = nn.Conv2d(channels, 1, kernel_size=1)


        # Layer 6
        self.process_layer6 = nn.Sequential(
            ConvBlock(196, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer6 = nn.Sequential(
            DeconvBlock(channels, channels, kernel=4, stride=2),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

        # Layer 5
        self.process_layer5 = nn.Sequential(
            ConvBlock(128, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer5 = nn.Sequential(
            DeconvBlock(channels*2, channels, kernel=4, stride=2),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

        # Layer 4
        self.process_layer4 = nn.Sequential(
            ConvBlock(96, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer4 = nn.Sequential(
            DeconvBlock(channels*2, channels, kernel=4, stride=2),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

        # Layer 3
        self.process_layer3 = nn.Sequential(
            ConvBlock(64, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer3 = nn.Sequential(
            DeconvBlock(channels*2, channels, kernel=4, stride=2),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

    def forward(self, input):
        features = self.extractor(input)

        output = self.process_layer6(features[5])
        output = self.upscale_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = self.upscale_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = self.upscale_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.upscale_layer3(output)
        return self.output_layer(output)


class PWCC_V3_64(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        channels = 64

        self.extractor = Extractor()
        if pretrained:
            self.extractor.load_state_dict(torch.load('chairs_extractor.pt'))

        self.output_layer = nn.Conv2d(channels, 1, kernel_size=1)


        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer6 = nn.Sequential(
            DeconvBlock(channels, channels, kernel=4, stride=2),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer5 = nn.Sequential(
            DeconvBlock(channels*2, channels, kernel=4, stride=2),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer4 = nn.Sequential(
            DeconvBlock(channels*2, channels, kernel=4, stride=2),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels, kernel=3, stride=1, dilation=1)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels*2, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

    def forward(self, input):
        features = self.extractor(input)

        output = self.process_layer6(features[5])
        output = self.upscale_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = self.upscale_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = self.upscale_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)
        return self.output_layer(output)


class PWCC_V3_128(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        channels = 128

        self.extractor = Extractor()
        if pretrained:
            self.extractor.load_state_dict(torch.load('chairs_extractor.pt'))

        self.output_layer = nn.Conv2d(channels, 1, kernel_size=1)


        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer6 = nn.Sequential(
            DeconvBlock(channels, channels, kernel=4, stride=2),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer5 = nn.Sequential(
            DeconvBlock(channels*2, channels, kernel=4, stride=2),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer4 = nn.Sequential(
            DeconvBlock(channels*2, channels, kernel=4, stride=2),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels, kernel=3, stride=1, dilation=1)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels*2, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

    def forward(self, input):
        features = self.extractor(input)

        output = self.process_layer6(features[5])
        output = self.upscale_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = self.upscale_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = self.upscale_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)
        return self.output_layer(output)


class PWCC_V3_32(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        channels = 32

        self.extractor = Extractor()
        if pretrained:
            self.extractor.load_state_dict(torch.load('chairs_extractor.pt'))

        self.output_layer = nn.Conv2d(channels, 1, kernel_size=1)


        # Layer 6
        channels6 = 196
        self.process_layer6 = nn.Sequential(
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
            ConvBlock(channels6, channels, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer6 = nn.Sequential(
            DeconvBlock(channels, channels, kernel=4, stride=2),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

        # Layer 5
        channels5 = 128
        self.process_layer5 = nn.Sequential(
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
            ConvBlock(channels5, channels, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer5 = nn.Sequential(
            DeconvBlock(channels*2, channels, kernel=4, stride=2),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

        # Layer 4
        channels4 = 96
        self.process_layer4 = nn.Sequential(
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
            ConvBlock(channels4, channels, kernel=3, stride=1, dilation=1)
        )

        self.upscale_layer4 = nn.Sequential(
            DeconvBlock(channels*2, channels, kernel=4, stride=2),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

        # Layer 3
        channels3 = 64
        self.process_layer3 = nn.Sequential(
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
            ConvBlock(channels3, channels, kernel=3, stride=1, dilation=1)
        )

        self.process_all = nn.Sequential(
            ConvBlock(channels*2, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
            ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
        )

    def forward(self, input):
        features = self.extractor(input)

        output = self.process_layer6(features[5])
        output = self.upscale_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = self.upscale_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = self.upscale_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)
        return self.output_layer(output)

class PWCC_V3_64_full(torch.nn.Module):
        def __init__(self, pretrained=True):
            super().__init__()

            channels = 64

            self.extractor = Extractor()
            if pretrained:
                self.extractor.load_state_dict(torch.load('chairs_extractor.pt'))

            self.output_layer = nn.Conv2d(channels, 1, kernel_size=1)

            # Layer 6
            channels6 = 196
            self.process_layer6 = nn.Sequential(
                ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
                ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
                ConvBlock(channels6, channels6, kernel=3, stride=1, dilation=1),
                ConvBlock(channels6, channels, kernel=3, stride=1, dilation=1)
            )

            self.upscale_layer6 = nn.Sequential(
                DeconvBlock(channels, channels, kernel=4, stride=2),
                ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
                ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
            )

            # Layer 5
            channels5 = 128
            self.process_layer5 = nn.Sequential(
                ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
                ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
                ConvBlock(channels5, channels5, kernel=3, stride=1, dilation=1),
                ConvBlock(channels5, channels, kernel=3, stride=1, dilation=1)
            )

            self.upscale_layer5 = nn.Sequential(
                DeconvBlock(channels * 2, channels, kernel=4, stride=2),
                ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
                ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
            )

            # Layer 4
            channels4 = 96
            self.process_layer4 = nn.Sequential(
                ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
                ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
                ConvBlock(channels4, channels4, kernel=3, stride=1, dilation=1),
                ConvBlock(channels4, channels, kernel=3, stride=1, dilation=1)
            )

            self.upscale_layer4 = nn.Sequential(
                DeconvBlock(channels * 2, channels, kernel=4, stride=2),
                ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
                ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
            )

            # Layer 3
            channels3 = 64
            self.process_layer3 = nn.Sequential(
                ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
                ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
                ConvBlock(channels3, channels3, kernel=3, stride=1, dilation=1),
                ConvBlock(channels3, channels, kernel=3, stride=1, dilation=1)
            )

            self.upscale_layer3 = nn.Sequential(
                DeconvBlock(channels * 2, channels, kernel=4, stride=2),
                ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
                ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
            )

            # Layer 2
            channels2 = 32
            self.process_layer2 = nn.Sequential(
                ConvBlock(channels2, channels2, kernel=3, stride=1, dilation=1),
                ConvBlock(channels2, channels2, kernel=3, stride=1, dilation=1),
                ConvBlock(channels2, channels2, kernel=3, stride=1, dilation=1),
                ConvBlock(channels2, channels, kernel=3, stride=1, dilation=1)
            )

            self.upscale_layer2 = nn.Sequential(
                DeconvBlock(channels * 2, channels, kernel=4, stride=2),
                ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
                ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
            )

            # Layer 1
            channels1 = 16
            self.process_layer1 = nn.Sequential(
                ConvBlock(channels1, channels1, kernel=3, stride=1, dilation=1),
                ConvBlock(channels1, channels1, kernel=3, stride=1, dilation=1),
                ConvBlock(channels1, channels1, kernel=3, stride=1, dilation=1),
                ConvBlock(channels1, channels, kernel=3, stride=1, dilation=1)
            )

            self.end_layer1 = nn.Sequential(
                DeconvBlock(channels * 2, channels, kernel=4, stride=2),
                ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
                ConvBlock(channels, channels, kernel=3, stride=1, dilation=1),
                ConvBlock(channels, channels, kernel=3, stride=1, dilation=1)
            )

        def forward(self, input):
            features = self.extractor(input)

            output = self.process_layer6(features[5])
            output = self.upscale_layer6(output)

            output = torch.cat((output, self.process_layer5(features[4])), 1)
            output = self.upscale_layer5(output)

            output = torch.cat((output, self.process_layer4(features[3])), 1)
            output = self.upscale_layer4(output)

            output = torch.cat((output, self.process_layer3(features[2])), 1)
            output = self.upscale_layer3(output)

            output = torch.cat((output, self.process_layer2(features[1])), 1)
            output = self.upscale_layer2(output)

            output = torch.cat((output, self.process_layer1(features[0])), 1)
            output = self.end_layer1(output)
            return self.output_layer(output)


class PWCC_V3_adapt(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.extractor = Extractor()
        if pretrained:
            self.extractor.load_state_dict(torch.load('chairs_extractor.pt'))


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

    def forward(self, input):
        features = self.extractor(input)

        output = self.process_layer6(features[5])
        output = self.upscale_layer6(output)

        output = torch.cat((output, self.process_layer5(features[4])), 1)
        output = self.upscale_layer5(output)

        output = torch.cat((output, self.process_layer4(features[3])), 1)
        output = self.upscale_layer4(output)

        output = torch.cat((output, self.process_layer3(features[2])), 1)
        output = self.process_all(output)
        return self.output_layer(output)