import torch
import numpy as np
import PIL
import PIL.Image

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, i):
        pair = self.pairs[i]
        tenFirst = torch.FloatTensor(np.ascontiguousarray(
            np.array(PIL.Image.open(pair.get_frames(0).get_image_path()))[:, :, ::-1].transpose(2, 0, 1).astype(
                np.float32) * (1.0 / 255.0)))
        tenSecond = torch.FloatTensor(np.ascontiguousarray(
            np.array(PIL.Image.open(pair.get_frames(1).get_image_path()))[:, :, ::-1].transpose(2, 0, 1).astype(
                np.float32) * (1.0 / 255.0)))

        return tenFirst, tenSecond

    def __len__(self):
        return len(self.pairs)