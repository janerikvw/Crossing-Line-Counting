import torch.nn as nn
import torch
from guided_filter.guided_filter import GuidedFilter as GuidedFilter

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from DDFlow_pytorch.model import Extractor, PWCNet

class ModelV2(torch.nn.Module):
    def __init__(self, apply_filter=False, load_pretrained=False):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                 torch.load(path).items()})



        self.apply_filter = apply_filter
        feature_channels = 64
        upscale_channels = 64
        back_layers = [256, 256, 128, 128, 64, 64]
        self.backend = self._make_layers(feature_channels, back_layers)

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(64, upscale_channels, 4, padding=3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(upscale_channels, upscale_channels, 4, padding=3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(upscale_channels, upscale_channels, 4, padding=3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Conv2d(upscale_channels, 1, kernel_size=1)
        self.output_filter = nn.Conv2d(upscale_channels, 1, kernel_size=3, padding=1)

        self.guided_filter = GuidedFilter(4, eps=1e-2)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features = True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density

    def cc_forward(self, features1, features2):
        input = features1[2]
        output = self.backend(input)
        output = self.upsampling(output)

        pred = self.output_layer(output)
        if self.apply_filter:
            filter = self.output_filter(output)
            output = self.guided_filter(filter, pred)
        else:
            output = pred
        return output

    def _make_layers(self, in_channels, cfg):
        layers = []
        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers)



class ModelV3(torch.nn.Module):
    def __init__(self, apply_filter=False, load_pretrained=False):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                 torch.load(path).items()})



        self.apply_filter = apply_filter
        feature_channels = 128
        upscale_channels = 64
        back_layers = [256, 256, 128, 128, 64, 64]
        self.backend = self._make_layers(feature_channels, back_layers)

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(back_layers[-1], upscale_channels, 4, padding=3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(upscale_channels, upscale_channels, 4, padding=3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(upscale_channels, upscale_channels, 4, padding=3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Conv2d(upscale_channels, 1, kernel_size=1)
        self.output_filter = nn.Conv2d(upscale_channels, 1, kernel_size=3, padding=1)

        self.guided_filter = GuidedFilter(4, eps=1e-2)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features = True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density

    def cc_forward(self, features1, features2):
        # input = features1[2]
        input = torch.cat((features1[2], features2[2]), dim=1)
        output = self.backend(input)
        output = self.upsampling(output)

        pred = self.output_layer(output)
        if self.apply_filter:
            filter = self.output_filter(output)
            output = self.guided_filter(filter, pred)
        else:
            output = pred
        return output

    def _make_layers(self, in_channels, cfg):
        layers = []
        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers)


class ModelV31(torch.nn.Module):
    def __init__(self, apply_filter=False, load_pretrained=False):
        super().__init__()

        self.fe_net = PWCNet()

        if load_pretrained == True:
            path = '../DDFlow_pytorch/network-chairs-things.pytorch'
            self.fe_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                 torch.load(path).items()})

        self.apply_filter = apply_filter
        feature_channels = 64
        upscale_channels = 64
        back_layers = [256, 256, 128, 128, 64, 64]
        self.backend = self._make_layers(feature_channels, back_layers)

        self.merger = nn.Sequential(
            nn.Conv2d(2*back_layers[-1], 2*upscale_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(2*upscale_channels, 2*upscale_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(2*upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True)
        )

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(upscale_channels, upscale_channels, 4, padding=3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(upscale_channels, upscale_channels, 4, padding=3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(upscale_channels, upscale_channels, 4, padding=3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(upscale_channels, upscale_channels, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Conv2d(upscale_channels, 1, kernel_size=1)
        self.output_filter = nn.Conv2d(upscale_channels, 1, kernel_size=3, padding=1)

        self.guided_filter = GuidedFilter(4, eps=1e-2)

    def forward(self, frame1, frame2):
        flow_fw, flow_bw, features1, features2 = self.fe_net.bidirection_forward(frame1, frame2, ret_features = True)
        density = self.cc_forward(features1, features2)
        return flow_fw, flow_bw, density

    def cc_forward(self, features1, features2):
        features = torch.cat((self.backend(features1[2]),
                           self.backend(features2[2])), dim=1)
        output = self.merger(features)
        output = self.upsampling(output)

        pred = self.output_layer(output)
        if self.apply_filter:
            filter = self.output_filter(output)
            output = self.guided_filter(filter, pred)
        else:
            output = pred
        return output

    def _make_layers(self, in_channels, cfg):
        layers = []
        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers)