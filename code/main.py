import torch
import sys

import datasets.factory as factory
import datasets.fudan as fudan
import LOI
import utils

from scipy import misc, io
from scipy.ndimage import rotate
import numpy as np

import datetime

from PIL import Image, ImageDraw
import PIL

scale = 1.0
orig_frame_width = 1920
orig_frame_height = 1080


print("--- LOADING TESTSET ---")
#train_pairs, test_pairs = factory.load_train_test_frame_pairs('')

train_pairs = fudan.load_video('data/Fudan/train_data/68').get_frame_pairs()
print("- Loaded {} pairs".format(len(train_pairs)))
print("--- DONE ---")

# Load counting model
cc_model = LOI.init_cc_model(weights_path='CSRNet/preA_fudanmodel_best.pth.tar', scale=scale)

# Load flow estimation model
# fe_model = LOI.init_fe_model(restore_model='DDFlow/Fudan/checkpoints/distillation_census_prekitty2/model-70000', scale=scale)

point1 = (550, 20)
point2 = (350, 700)
loi_model = LOI.init_regionwise_loi(point1, point2, orig_img_width=orig_frame_width, orig_img_height=orig_frame_height, scale=scale, shrink=0.92, orig_width=35)

total_left = 0
total_right = 0

for i, pair in enumerate(train_pairs[0:2]):
    print_i = '{:05d}'.format(i+1)
    print("{}/{}".format(print_i, len(train_pairs)))

    tbegin = datetime.datetime.now()

    # Run counting model on frame A
    cc_output = LOI.run_cc_model(cc_model, pair.get_frames()[0])

    print("Crowd Counting miliseconds: {}".format(int((datetime.datetime.now() - tbegin).total_seconds() * 1000)))
    tbegin = datetime.datetime.now()

    # Run flow estimation model on frame pair
    # fe_output, fe_output_color = LOI.run_fe_model(fe_model, pair)

    print("Flow Estimator miliseconds: {}".format(int((datetime.datetime.now() - tbegin).total_seconds() * 1000)))
    tbegin = datetime.datetime.now()

    # Run the merger for the crowdcounting and flow estimation model
    # In code mention the papers based on
    # loi_output = LOI.regionwise_loi(loi_model, cc_output, fe_output)

    print("LOI miliseconds: {}".format(int((datetime.datetime.now() - tbegin).total_seconds() * 1000)))

    # Sum earlier LOI's
    to_right = sum(loi_output[1])
    to_left = sum(loi_output[0])
    total_left += to_left
    total_right += to_right
    totals = [total_left, total_right]

    # Load original image or demo
    total_image = pair.get_frames()[0].get_image().convert("L").convert("RGBA")

    tbegin = datetime.datetime.now()

    # Save original flow image
    misc.imsave('results/video2/orig_flow_{}.png'.format(print_i), fe_output_color[0])

    # Transform flow and merge with original image
    flow_img = fe_output_color[0] * 255.0
    flow_img = flow_img.astype(np.uint8)
    flow_img = Image.fromarray(flow_img, 'RGB')
    flow_img = utils.white_to_transparency_gradient(flow_img)
    total_image = Image.alpha_composite(total_image, flow_img)

    # Load and save original CC model
    cc_img = Image.fromarray(cc_output * 255.0 / cc_output.max())
    cc_img = cc_img.convert("L")
    cc_img.save('results/video2/csr_{}.png'.format(print_i))

    # Transform CC model and merge with original image for demo
    cc_img = np.zeros((cc_output.shape[0], cc_output.shape[1], 4))
    cc_img = cc_img.astype(np.uint8)
    cc_img[:, :, 3] = 255 - (cc_output * 255.0 / cc_output.max())
    cc_img[cc_img > 2] = cc_img[cc_img > 2] * 0.5
    cc_img = Image.fromarray(cc_img, 'RGBA')
    total_image = Image.alpha_composite(total_image, cc_img)

    # Save merged image
    total_image.save('results/video2/combined_{}.png'.format(print_i))

    print("Combining CC/FE and orig: {}".format(int((datetime.datetime.now() - tbegin).total_seconds() * 1000)))
    tbegin = datetime.datetime.now()


    # Generate the demo output for clear view on what happens
    img = pair.get_frames()[0].get_image().convert("RGB")
    img = img.resize((int(orig_frame_width * scale), int(orig_frame_height * scale)))
    scaled_point1 = (point1[0] * scale, point1[1] * scale)
    scaled_point2 = (point2[0] * scale, point2[1] * scale)
    utils.image_add_region_lines(img, scaled_point1, scaled_point2, loi_output, shrink=0.92, width=35*scale)

    # Crop and add information at the bottom of the screen
    img = img.crop((point2[0] - 70, point1[1] - 100, point1[0] + 70, point2[1] + 100))
    utils.add_total_information(img, loi_output, totals)

    img.save('results/video2/orig_{}.png'.format(print_i))

    print("Demo saving miliseconds: {}".format(int((datetime.datetime.now() - tbegin).total_seconds() * 1000)))
