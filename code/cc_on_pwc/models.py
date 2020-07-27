import torch.nn as nn
import torch

class ModelV1(torch.nn.Module):
    def __init__(self):
        super().__init__()

        feature_channels = 64
        back_layers = [256, 256, 128, 128, 64, 64]
        self.backend = self._make_layers(feature_channels, back_layers)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, features):
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

