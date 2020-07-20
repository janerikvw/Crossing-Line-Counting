class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def norm_to_img(tensor):
    if len(tensor.shape):
        tensor = tensor.unsqueeze(0)

    channel_max = tensor.max(axis=2)[0].max(axis=1)[0]
    return tensor / channel_max

import datetime
class sTimer():
    def __init__(self, name):
        self.start = datetime.datetime.now()
        self.name = name

    def show(self, printer=True):
        ms = int((datetime.datetime.now() - self.start).total_seconds() * 1000)
        if printer:
            print("{}: {}ms".format(self.name, ms))

        return ms