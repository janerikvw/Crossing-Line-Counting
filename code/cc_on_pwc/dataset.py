import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch.nn.functional as F

import math
import random

import numpy as np
import PIL
import PIL.Image

# Simple data which loads a full
class BasicDataset(Dataset):
    def __init__(self, frames, density_type=None, resize_diff=1.0, augmentation=False):
        self.frames = frames
        self.density_type = density_type
        self.resize_diff = resize_diff
        self.augmentation = augmentation

        # self.transform = T.Compose([T.ToTensor(),
        #                             T.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])])

        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=[0.410824894905, 0.370634973049, 0.359682112932],
                                                std=[0.278580576181, 0.26925137639, 0.27156367898])])

    def __getitem__(self, item):
        frame = self.frames[item]
        # Load the image and normalize
        frame_img = PIL.Image.open(frame.get_image_path()).convert('RGB')
        frame_img = self.transform(frame_img)

        # Load the density map and add single layer for PyTorch
        frame_density = torch.FloatTensor(frame.get_density(self.density_type))
        frame_density.unsqueeze_(0)

        # Resize
        int_preprocessed_width = int(math.floor(math.ceil(frame_img.shape[2] / 64) * 64))
        int_preprocessed_height = int(math.floor(math.ceil(frame_img.shape[1] / 64) * 64))

        factor = int(int_preprocessed_width * int_preprocessed_height) \
                 / int(frame_density.shape[1] * frame_density.shape[2])
        # Resize to get a size which fits into the network
        frame_img.unsqueeze_(0)
        frame_density.unsqueeze_(0)
        frame_img = F.interpolate(input=frame_img,
                                  size=(int_preprocessed_height, int_preprocessed_width),
                                  mode='bilinear', align_corners=False)
        frame_density = F.interpolate(input=frame_density,
                                      size=(int_preprocessed_height, int_preprocessed_width),
                                      mode='bilinear', align_corners=False) / factor
        frame_img.squeeze_(0)
        frame_density.squeeze_(0)

        if self.augmentation:
            divider = 2
            crop_size = (int(frame_img.shape[1] / divider / 64) * 64, int(frame_img.shape[2] / divider / 64) * 64)

            dx = int(random.random() * (frame_img.shape[1] - crop_size[0]))
            dy = int(random.random() * (frame_img.shape[2] - crop_size[1]))

            frame_img = frame_img[:, dx:dx+crop_size[0], dy:dy+crop_size[1]]
            frame_density = frame_density[:, dx:dx+crop_size[0], dy:dy+crop_size[1]]

        # Do augmentation
        if self.augmentation:
            if np.random.random() > 0.5:
                frame_img = frame_img.flip(2)
                frame_density = frame_density.flip(2)

        return frame_img, frame_density

    def __len__(self):
        return len(self.frames)
