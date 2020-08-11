import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

import math

import numpy as np
import PIL
import PIL.Image

class DRNetDataset(Dataset):
    def __init__(self, frames, density_type=None, patch_size=None):
        self.frames = frames
        self.density_type = density_type
        self.patch_size = patch_size

    def __getitem__(self, item):
        frame = self.frames[item]
        # Load the image and normalize
        frame_img = torch.FloatTensor(np.ascontiguousarray(
            np.array(PIL.Image.open(frame.get_image_path()).convert('RGB'))[:, :, ::-1].transpose(2, 0, 1).astype(
                np.float32) * (1.0 / 255.0)))

        frame_density = torch.FloatTensor(frame.get_density(self.density_type))
        frame_density *= 100.0

        frame_density.unsqueeze_(0)

        # Random crop to patch, so it is training
        if self.patch_size:
            int_preprocessed_width = int(math.floor(math.ceil(frame_img.shape[2] / 64.0) * 64.0))
            int_preprocessed_height = int(math.floor(math.ceil(frame_img.shape[1] / 64.0) * 64.0))

            frame_img.unsqueeze_(0)
            frame_density.unsqueeze_(0)

            # Resize to get a size which fits into the network
            frame_img = torch.nn.functional.interpolate(input=frame_img,
                                                     size=(int_preprocessed_height, int_preprocessed_width),
                                                     mode='bilinear', align_corners=False)

            # Resize to get a size which fits into the network
            frame_density = torch.nn.functional.interpolate(input=frame_density,
                                                        size=(int_preprocessed_height, int_preprocessed_width),
                                                        mode='bilinear', align_corners=False)

            frame_img.squeeze_(0)
            frame_density.squeeze_(0)

            # i, j, h, w = T.RandomCrop.get_params(
            #     frame_img, output_size=self.patch_size)
            #
            # frame_img = frame_img[:, i:i+h, j:j+w]
            # frame_density = frame_density[:, i:i + h, j:j + w]

            # Augmentation
            # if np.random.random() > 0.5:
            #     frame_img = frame_img.flip(2)
            #     frame_density = frame_density.flip(2)

        return frame_img, frame_density

    def __len__(self):
        return len(self.frames)
