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