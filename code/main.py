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

train_pairs = fudan.load_video('data/Fudan/train_data/68').get_frame_pairs()
print("- Loaded {} pairs".format(len(train_pairs)))
print("--- DONE ---")

# Load counting model
cc_model = LOI.init_cc_model(weights_path='CSRNet/preA_fudanmodel_best.pth.tar')

# Load flow estimation model
fe_model = LOI.init_fe_model()

point1 = (550, 20)
point2 = (350, 700)
loi_region_info = LOI.init_regionwise_loi(point1, point2, shrink=0.88)

total_left = 0
total_right = 0

for i, pair in enumerate(train_pairs):
    print_i = '{:05d}'.format(i+1)
    
    print("{}/{}".format(print_i, len(train_pairs)))
    # Run counting model on frame A
    cc_output = LOI.run_cc_model(cc_model, pair.get_frames()[0])
    img = Image.fromarray(cc_output * 255.0 / cc_output.max())
    img = img.convert("L")
    img.save('results/video2/csr_{}.png'.format(print_i))

    # Run flow estimation model on frame pair
    fe_output, fe_output_color = LOI.run_fe_model(fe_model, pair)
    misc.imsave('results/video2/flow_{}.png'.format(print_i), fe_output_color[0])

    # Run the merger for the crowdcounting and flow estimation model
    # In code mention the papers based on
    loi_output = LOI.regionwise_loi(cc_output, fe_output, loi_region_info)

    # Sum earlier LOI's
    to_right = sum(loi_output[1])
    to_left = sum(loi_output[0])
    total_left += to_left
    total_right += to_right
    totals = [total_left, total_right]

    img = pair.get_frames()[0].get_image().convert("RGB")
    LOI.image_add_region_lines(img, point1, point2, loi_output)


    img = img.crop((point2[0] - 70, point1[1] - 100, point1[0] + 70, point2[1] + 100))
    LOI.add_total_information(img, loi_output, totals)

    # LOI.image_add_region_lines(img, point1, point2)

    img.save('results/video2/orig_{}.png'.format(print_i))