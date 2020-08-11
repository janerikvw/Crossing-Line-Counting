#######
## TODO: Update this shit for correct testing!!!
#######
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from torchvision import transforms
import scipy
import json

import torch

import torchvision.transforms.functional as F
from matplotlib import cm as CM
from model import CSRNet
from PIL import Image, ImageDraw

from scipy.ndimage.interpolation import zoom


from torchvision import datasets, transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])


root = '/home/jvwoerden/Thesis/code/data/ShanghaiTech'

#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_A_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)


model = CSRNet()

model = model.cuda()
checkpoint = torch.load('ShanghaiA-sigma5checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

mae = 0
result_dir = 'test'
for i in xrange(len(img_paths)):
    # img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))
    #
    # img[0,:,:]=img[0,:,:]-92.8207477031
    # img[1,:,:]=img[1,:,:]-95.2757037428
    # img[2,:,:]=img[2,:,:]-104.877445883
    # img = img.cuda()
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    groundtruth = np.asarray(gt_file['density'])
    output = model(img.unsqueeze(0))
    cc_output = output.detach().cpu()
    i_mae = abs(cc_output.sum().numpy()-np.sum(groundtruth))


    #test_gt_file = np.load(img_paths[i].replace('.jpg','.npy'))
    #print(np.sum(gt_file))
    # test_gt_file = Image.fromarray(groundtruth * 255.0 / groundtruth.max())
    # test_gt_file = test_gt_file.convert("L")
    # test_gt_file.save(os.path.join(result_dir, 'gt_{}.png').format(i))
    #
    # cc_output = cc_output.data.numpy().squeeze()
    # cc_output = zoom(cc_output, zoom=8.0) / 64.
    #
    # cc_img = Image.fromarray(cc_output * 255.0 / cc_output.max())
    # cc_img = cc_img.convert("L")
    # cc_img.save(os.path.join(result_dir, 'result_{}.png').format(i))
    # Image.open(img_paths[i]).convert('RGB').save(os.path.join(result_dir, 'orig_{}.png').format(i))

    print i, mae, i_mae, np.sum(groundtruth), output.detach().cpu().sum().numpy()
    if i > 3:
        break
