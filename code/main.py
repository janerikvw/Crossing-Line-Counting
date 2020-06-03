
import torch
import sys
import os
import tensorflow as tf

import datasets.factory as factory
import datasets.fudan as fudan
import datasets.arenapeds as arenapeds
import LOI
import utils

from scipy import misc, io
from scipy.ndimage import rotate
import numpy as np

import datetime

from PIL import Image, ImageDraw
import PIL

# Configurable params
scale = 2/3.
orig_frame_width = 1920
orig_frame_height = 1080

orig_point1 = (550, 20)
orig_point2 = (350, 700)
orig_line_width = 50  # Widest region
orientation_shrink = 0.92  # Shrinking per region moving away from the camera (To compensate for orientation)

result_dir = 'results/new_video'

# Create dir if not exists
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

point1, point2, frame_width, frame_height, line_width = utils.scale_params(orig_point1,
                                                               orig_point2,
                                                               orig_frame_width,
                                                               orig_frame_height,
                                                               orig_line_width,
                                                               scale=scale)

print("--- LOADING TESTSET ---")
#train_pairs, test_pairs = factory.load_train_test_frame_pairs('')



train_pairs = fudan.load_video('data/Fudan/train_data/68').get_frame_pairs()

# all_videos = arenapeds.load_all_videos('data/ArenaPeds/smaller_train_images')
# train_pairs = all_videos[all_videos.keys()[1]].get_frame_pairs()
print("- Loaded {} pairs".format(len(train_pairs)))
print("--- DONE ---")


# Load counting model
cc_model = LOI.init_cc_model(weights_path='CSRNet/V2_PreB_Fudanmodel_best.pth.tar', img_width=frame_width, img_height=frame_height)
#drnet_model = LOI.init_drnet_model()

# Load flow estimation model
fe_model = LOI.init_fe_model(restore_model='DDFlow/Fudan/checkpoints/distillation_census_prekitty2/model-70000', img_width=frame_width, img_height=frame_height)

loi_model = LOI.init_regionwise_loi(point1, point2, img_width=orig_frame_width, img_height=orig_frame_height, shrink=orientation_shrink, line_width=line_width)

total_left = 0
total_right = 0

for i, pair in enumerate(train_pairs):
    print_i = '{:05d}'.format(i+1)
    print("{}/{}".format(print_i, len(train_pairs)))

    tbegin = datetime.datetime.now()

    frame1_img = Image.open(pair.get_frames()[0].get_image_path()).convert('RGB').resize((frame_width, frame_height))
    # frame2_img = Image.open(pair.get_frames()[0].get_image_path()).convert('RGB').resize((frame_width, frame_height))

    print("Loading images: {}ms".format(int((datetime.datetime.now() - tbegin).total_seconds() * 1000)))
    tbegin = datetime.datetime.now()

    # Run counting model on frame A
    cc_output = LOI.run_cc_model(cc_model, frame1_img.copy())
    # cc_output = LOI.run_drnet_model(drnet_model, pair.get_frames()[0])

    print("Crowd Counting: {}ms".format(int((datetime.datetime.now() - tbegin).total_seconds() * 1000)))
    tbegin = datetime.datetime.now()

    # Run flow estimation model on frame pair
    fe_output, fe_output_color = LOI.run_fe_model(fe_model, pair)  # Optimize by frame1_img.copy(), frame2_img.copy()

    print("Flow Estimator: {}ms".format(int((datetime.datetime.now() - tbegin).total_seconds() * 1000)))
    tbegin = datetime.datetime.now()

    # Run the merger for the crowdcounting and flow estimation model
    # In code mention the papers based on
    loi_output = LOI.regionwise_loi(loi_model, cc_output, fe_output)

    print("LOI: {}ms".format(int((datetime.datetime.now() - tbegin).total_seconds() * 1000)))

    # Sum earlier LOI's
    to_right = sum(loi_output[1])
    to_left = sum(loi_output[0])
    total_left += to_left
    total_right += to_right
    totals = [total_left, total_right]

    # Load original image or demo
    total_image = frame1_img.copy().convert("L").convert("RGBA")

    tbegin = datetime.datetime.now()

    # Load and save original CC model
    cc_img = Image.fromarray(cc_output * 255.0 / cc_output.max())
    cc_img = cc_img.convert("L")
    cc_img.save(os.path.join(result_dir, 'crowd_{}.png').format(print_i))

    # Transform CC model and merge with original image for demo
    cc_img = np.zeros((cc_output.shape[0], cc_output.shape[1], 4))
    cc_img = cc_img.astype(np.uint8)
    cc_img[:, :, 3] = 255 - (cc_output * 255.0 / cc_output.max())
    cc_img[cc_img > 2] = cc_img[cc_img > 2] * 0.4
    cc_img = Image.fromarray(cc_img, 'RGBA')
    total_image = Image.alpha_composite(total_image, cc_img)

    # Save original flow image
    misc.imsave(os.path.join(result_dir, 'flow_{}.png'.format(print_i)), fe_output_color[0])

    # Transform flow and merge with original image
    flow_img = fe_output_color[0] * 255.0
    flow_img = flow_img.astype(np.uint8)
    flow_img = Image.fromarray(flow_img, 'RGB')
    flow_img = utils.white_to_transparency_gradient(flow_img)
    total_image = Image.alpha_composite(total_image, flow_img)

    # Save merged image
    total_image.save(os.path.join(result_dir, 'combined_{}.png'.format(print_i)))
    img = total_image.convert('RGB')
    # Generate the demo output for clear view on what happens
    utils.image_add_region_lines(img, point1, point2, loi_output, shrink=orientation_shrink, width=line_width)

    # Crop and add information at the bottom of the screen
    img = img.crop((point2[0] - 100, point1[1] - 15, point1[0] + 100, point2[1] + 60))
    utils.add_total_information(img, loi_output, totals)

    img.save(os.path.join(result_dir, 'line_{}.png'.format(print_i)))

    print("Merge results and save demo image: {}ms".format(int((datetime.datetime.now() - tbegin).total_seconds() * 1000)))
