import torch
import sys

import datasets.factory as factory
import datasets.fudan as fudan
import LOI

from scipy import misc, io
from scipy.ndimage import rotate
import numpy as np

from PIL import Image, ImageDraw

print("--- LOADING TESTSET ---")
#train_pairs, test_pairs = factory.load_train_test_frame_pairs('')

train_pairs = fudan.load_video('data/Fudan/train_data/72').get_frame_pairs()
print("- Loaded {} pairs".format(len(train_pairs)))
print("--- DONE ---")

# Load counting model
cc_model = LOI.init_cc_model()

# Load flow estimation model
#fe_model = LOI.init_fe_model()

point1 = (320, 400)
point2 = (1250, 460)
loi_region_info = LOI.init_regionwise_loi(point1, point2)

for i,pair in enumerate(train_pairs):

    print("{}/{}".format(i, len(train_pairs)))
    # Run counting model on frame A
    cc_output = LOI.run_cc_model(cc_model, pair.get_frames()[0])

    # Run flow estimation model on frame pair
    #fe_output = LOI.run_fe_model(fe_model, pair)

    # Run the merger for the crowdcounting and flow estimation model
    # In code mention the papers based on
    fe_output = None
    loi_output = LOI.regionwise_loi(cc_output, fe_output, loi_region_info)
    #
    # img = Image.fromarray(cc_output * 255.0 / cc_output.max())
    # img = img.convert("L")
    # img.save('results/csr.png')

    img = pair.get_frames()[0].get_image().convert("RGB")

    LOI.image_add_region_lines(img, point1, point2, loi_output)

    img = img.crop((point1[0]-50, point1[1]-200, point2[0]+50, point2[1]+200))

    img.save('results/video/r_{}.png'.format(i))

    # misc.imsave('results/flow_est_.png', fe_output[1][0])