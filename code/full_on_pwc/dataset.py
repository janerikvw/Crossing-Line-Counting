import torch
import numpy as np
import PIL
import PIL.Image
import utils
import random

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, density_type=None, augmentation=False):
        self.pairs = pairs
        self.density_type = density_type
        self.augmentation = augmentation

    def __getitem__(self, i):
        pair = self.pairs[i]

        # Load the image and normalize
        frame1 = torch.FloatTensor(np.ascontiguousarray(
            np.array(PIL.Image.open(pair.get_frames(0).get_image_path()))[:, :, ::-1].transpose(2, 0, 1).astype(
                np.float32) * (1.0 / 255.0)))
        frame2 = torch.FloatTensor(np.ascontiguousarray(
            np.array(PIL.Image.open(pair.get_frames(1).get_image_path()))[:, :, ::-1].transpose(2, 0, 1).astype(
                np.float32) * (1.0 / 255.0)))

        density = torch.FloatTensor(pair.get_frames(0).get_density(self.density_type))
        density = density.unsqueeze_(0)

        # Crop to 1/2 of the image
        if self.augmentation:
            crop_size = (int(frame1.shape[1] / 2), int(frame1.shape[2] / 2))

            dx = int(random.random() * frame1.shape[1] * 1. / 2)
            dy = int(random.random() * frame1.shape[2] * 1. / 2)

            frame1 = frame1[:, dx:dx+crop_size[0], dy:dy+crop_size[1]]
            frame2 = frame2[:, dx:dx + crop_size[0], dy:dy + crop_size[1]]
            density = density[:, dx:dx+crop_size[0], dy:dy+crop_size[1]]

        # Flip sometimes
        if self.augmentation:
            if random.random() > 0.5:
                frame1 = frame1.flip(2)
                frame2 = frame2.flip(2)
                density = density.flip(2)

        return frame1, frame2, density

    def __len__(self):
        return len(self.pairs)
