import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import cv2

class listDataset(Dataset):
    def __init__(self, root, density_name=None, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        # TODO: Fix this later for nice length handling
        if train:
            if len(root) < 500:
                root = root * 4

            random.shuffle(root)

            if len(root) > 1500:
                root = root[0:1500]

        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.density_name = density_name
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        
        frame_obj = self.lines[index]

        img = frame_obj.get_image().convert('RGB')
        target = frame_obj.get_density(self.density_name)

        target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64

        if self.transform is not None:
            img = self.transform(img)
            
        return img,target
