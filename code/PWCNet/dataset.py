import torch
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, i):
        elem = self.pairs[i]
        

    def __len__(self):
        return len(self.pairs)