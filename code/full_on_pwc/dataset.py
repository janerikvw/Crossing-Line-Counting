import torch
import torch.nn.functional as F
import numpy as np
import PIL
import PIL.Image
import utils
import random

import density_filter


def image_augmentation(frame, cropping_size, crop_mag, crop_loc_rand, flip):
    crop_size = (int(frame.shape[1] / crop_mag), int(frame.shape[2] / crop_mag))
    crop_loc = (int(crop_loc_rand[0] * (frame.shape[1] - crop_size[0])),
                int(crop_loc_rand[1] * (frame.shape[2] - crop_size[1])))
    patch_size = (int(frame.shape[1] / cropping_size), int(frame.shape[2] / cropping_size))
    frame = frame[:, crop_loc[0]:crop_loc[0] + crop_size[0], crop_loc[1]:crop_loc[1] + crop_size[1]]

    if frame.shape[1] != crop_size[0] or frame.shape[2] != crop_size[1]:
        print("Wrong patch")
        print("size", crop_size)
        print("loc", crop_loc)
        print(crop_mag, crop_loc_rand)
        exit()

    if cropping_size != crop_mag:
        frame.unsqueeze_(0)
        frame = F.interpolate(input=frame, size=patch_size, mode='bicubic', align_corners=False)
        frame.squeeze_(0)

    if flip:
        frame = frame.flip(2)

    return frame


def generate_density(frame, density_type):
    if density_type[:5] != 'fixed':
        print("Can only handle fixed densities right now")
        exit()

    density_size = int(density_type[6:])

    return density_filter.gaussian_filter_fixed_density(frame, density_size)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, args, augmentation=False):
        self.pairs = pairs
        self.args = args
        self.density_type = args.density_model
        self.augmentation = augmentation
        if args.resize_patch == 'on':
            self.resize_patch = True
        else:
            self.resize_patch = False

        if args.dataset == 'ucsd':
            self.cropping_size = 1
        elif args.dataset == 'tub':
            self.cropping_size = 2
        else:
            self.cropping_size = 2

    def __getitem__(self, i):
        pair = self.pairs[i]

        # Load the image and normalize
        frame1 = torch.FloatTensor(np.ascontiguousarray(
            np.array(PIL.Image.open(pair.get_frames(0).get_image_path()))[:, :, ::-1].transpose(2, 0, 1).astype(
                np.float32) * (1.0 / 255.0)))
        frame2 = torch.FloatTensor(np.ascontiguousarray(
            np.array(PIL.Image.open(pair.get_frames(1).get_image_path()))[:, :, ::-1].transpose(2, 0, 1).astype(
                np.float32) * (1.0 / 255.0)))

        #density = torch.FloatTensor(pair.get_frames(0).get_density(self.density_type))
        density = torch.FloatTensor(generate_density(pair.get_frames(0), self.density_type))
        density = density.unsqueeze_(0)

        density2 = torch.FloatTensor(generate_density(pair.get_frames(1), self.density_type))
        density2 = density2.unsqueeze_(0)

        # Crop to 1/2 of the image. (With some resizing for augmentation) and flip sometimes
        if self.augmentation:
            if self.resize_patch:
                crop_mag = random.uniform(0.875 * self.cropping_size, 1.25 * self.cropping_size)
            else:
                crop_mag = self.cropping_size

            r = 9 # random.randrange(9)
            if r == 5:
                crop_loc_rand = (0.0, 0.0)
            elif r == 6:
                crop_loc_rand = (1.0, 0.0)
            elif r == 7:
                crop_loc_rand = (0.0, 1.0)
            elif r == 8:
                crop_loc_rand = (1.0, 1.0)
            else:
                crop_loc_rand = (random.random(), random.random())

            flip = random.random() > 0.5

            frame1 = image_augmentation(frame1, self.cropping_size, crop_mag, crop_loc_rand, flip)
            frame2 = image_augmentation(frame2, self.cropping_size, crop_mag, crop_loc_rand, flip)
            density = image_augmentation(density, self.cropping_size, crop_mag, crop_loc_rand, flip)
            density2 = image_augmentation(density2, self.cropping_size, crop_mag, crop_loc_rand, flip)

        return frame1, frame2, density, density2

    def __len__(self):
        return len(self.pairs)
