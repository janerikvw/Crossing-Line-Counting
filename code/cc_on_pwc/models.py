import torch.nn as nn
import torch
from torchvision import models

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from DDFlow_pytorch.model import Extractor
from guided_filter.guided_filter import GuidedFilter as GuidedFilter


class ModelV1(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.extractor = Extractor()

        feature_channels = 64
        back_layers = [256, 256, 128, 128, 64, 64]
        self.backend = self._make_layers(feature_channels, back_layers)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, input):
        features = self.extractor(input)
        input = features[2]
        output = self.backend(input)
        output = self.output_layer(output)
        return output


    def _make_layers(self, in_channels, cfg):
        layers = []
        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers)


class ModelV2(torch.nn.Module):
    def __init__(self, apply_filter=False):
        super().__init__()

        self.extractor = Extractor()
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

    def forward(self, input):
        features = self.extractor(input)
        input = features[2]
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


class ResCSRNet(nn.Module):
    def __init__(self):
        super(ResCSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)

        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        res = models.resnet50(pretrained=True)
        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        )

    def forward(self, x):
        x = self.frontend(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ResCSRNetV2(nn.Module):
    def __init__(self):
        super(ResCSRNetV2, self).__init__()
        # self.backend_feat = [512, 512, 512, 256, 128, 64]
        # self.backend = make_layers(self.backend_feat, in_channels=1024, dilation=True)
        # self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self.de_pred = nn.Sequential(Conv2d(1024, 128, 1, same_padding=True, NL='relu'),
                                     Conv2d(128, 1, 1, same_padding=True, NL='relu'))

        res = models.resnet50(pretrained=True)
        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        )

        self.reslayer_3 = make_res_layer(Bottleneck, 256, 6, stride=1)
        self.reslayer_3.load_state_dict(res.layer3.state_dict())

    def forward(self, x):
        x = self.frontend(x)
        x = self.reslayer_3(x)
        x = self.de_pred(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ResCSRNetV3(nn.Module):
    def __init__(self):
        super(ResCSRNet, self).__init__()
        # self.backend_feat = [512, 512, 512, 256, 128, 64]
        # self.backend = make_layers(self.backend_feat, in_channels=1024, dilation=True)
        # self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self.de_pred = nn.Sequential(Conv2d(1024, 128, 1, same_padding=True, NL='relu'),
                                     Conv2d(128, 1, 1, same_padding=True, NL='relu'))

        res = models.resnet50(pretrained=True)
        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        )

        self.reslayer_3 = make_res_layer(Bottleneck, 256, 6, stride=1)
        self.reslayer_3.load_state_dict(res.layer3.state_dict())

    def forward(self, x):
        x = self.frontend(x)
        x = self.reslayer_3(x)
        x = self.de_pred(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_res_layer(block, planes, blocks, stride=1):

    downsample = None
    inplanes=512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
