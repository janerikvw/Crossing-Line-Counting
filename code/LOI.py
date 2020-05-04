import numpy as np
import torch
import math

import datasets

from utils import *

import scipy.misc
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import zoom

from CSRNet.model import CSRNet
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw

import tensorflow as tf
from DDFlow.network import pyramid_processing
from DDFlow.flowlib import flow_to_color

import datetime


# Intialize the regionwise LOI. Returns all the information required to execute the regionwise LOI optimally.
# The initializer generates all the reqions around the LOI and the masks for extracting the data from the CC and FE.
def init_regionwise_loi(point1, point2, img_width=1920, img_height=1080, width=50, cregions=5, shrink=0.90):
    center = (img_width / 2, img_height / 2)
    rotate_angle = math.degrees(math.atan((point2[1] - point1[1]) / float(point2[0] - point1[0])))

    # Select regions on a rotated image, so all the regions can be used to create simple masks
    # regions = select_regions(rotate_point(point1, -rotate_angle, center), rotate_point(point2, -rotate_angle, center),
    #                          width=width, regions=cregions, shrink=shrink)

    # Generate the original regions as well for later usage
    regions = select_regions(point1, point2,
                             width=width, regions=cregions, shrink=shrink)

    # Generate per region a mask which can be used to extract the crowd counting and flow estimation
    masks = ([], [])
    for i, small_regions in enumerate(regions):
        for o, region in enumerate(small_regions):
            mask = region_to_mask2(region, rotate_angle, center)
            masks[i].append(mask)

    return regions, rotate_angle, center, masks


# Combine both the crowd counting and the flow estimation with a regionwise method
# Returns per region how many people are crossing the line from that side.
def regionwise_loi(loi_info, counting_result, flow_result):
    regions, rotate_angle, center, masks = loi_info

    sums = ([], [])

    flow_result = np.squeeze(flow_result['full_res'])

    # @TODO: small_regions to something more correct
    for i, side_regions in enumerate(regions):
        for o, region in enumerate(side_regions):
            mask = masks[i][o]

            # Get which part of the mask contains the actual mask
            # This massively improves the speed of the model
            points = np.array(region[0:4])
            min_x = np.min(points[:, 0])
            max_x = np.max(points[:, 0])
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])

            cropped_mask = mask[min_y:max_y, min_x:max_x]

            # Use cropped mask on crowd counting result
            cc_part = cropped_mask * counting_result[min_y:max_y, min_x:max_x]

            # Project the flow estimation output on a line perpendicular to the LOI,
            # so we can calculate if the people are approach/leaving the LOI.
            direction = np.array([region[1][0] - region[2][0], region[1][1] - region[2][1]]).astype(
                np.float32)
            direction = direction / np.linalg.norm(direction)

            # Crop so only cropped area gets projected
            part_flow_result = flow_result[min_y:max_y, min_x:max_x]
            perp = np.sum(np.multiply(part_flow_result, direction), axis=2)

            # Get the FE region with mask
            fe_part = cropped_mask * perp

            # Get all the movement towards the line
            total_pixels = masks[i][o].sum()
            threshold = 1
            towards_pixels = fe_part > threshold

            total_crowd = cc_part.sum()

            # Too remove some noise
            if towards_pixels.sum() == 0 or total_crowd < 0:
                sums[i].append(0.0)
                continue

            towards_avg = fe_part[towards_pixels].sum() / float(total_pixels) # float(towards_pixels.sum())
            away_pixels = fe_part < -threshold
            # away_avg = fe_part[away_pixels].sum() / away_pixels.sum()

            pixel_percentage_towards = towards_pixels.sum() / float(away_pixels.sum()+towards_pixels.sum())
            crowd_towards = total_crowd * pixel_percentage_towards

            # Divide the average
            percentage_over = towards_avg / region[4]

            sums[i].append(crowd_towards * percentage_over)

    return sums


# Initialize the crowd counting model
def init_cc_model(weights_path = 'CSRNet/fudan_only_model_best.pth.tar', scale=1.0):
    print("--- LOADING CSRNET ---")
    print("- Initialize architecture")
    cc_model = CSRNet()
    print("- Architecture to GPU")
    cc_model = cc_model.cuda()
    print("- Loading weights")
    checkpoint = torch.load(weights_path)
    print("- Weights to GPU")
    cc_model.load_state_dict(checkpoint['state_dict'])
    print("--- DONE ---")
    return cc_model, scale


# Run the crowd counting model on frame A
def run_cc_model(cc_info, frame):
    cc_model, scale = cc_info

    # Load the image and normalize the image
    img = 255.0 * F.to_tensor(Image.open(frame.get_image_path()).convert('RGB'))
    img[0, :, :] = img[0, :, :] - 92.8207477031
    img[1, :, :] = img[1, :, :] - 95.2757037428
    img[2, :, :] = img[2, :, :] - 104.877445883

    # Run the model on the GPU
    img = img.cuda()
    cc_output = cc_model(img.unsqueeze(0))

    # Back to numpy and resize to original size
    cc_output = cc_output.detach().cpu().data.numpy().squeeze()
    cc_output = zoom(cc_output, zoom=8.0) / 64.

    return cc_output


# Initialize the flow estimation model
def init_fe_model(restore_model='DDFlow/Fudan/checkpoints/distillation_census_prekitty2/model-70000', scale=1.0):
    print("--- LOADING DDFLOW ---")

    print("- Load architecture")
    frame_img1 = tf.placeholder(tf.float32, [1, None, None, 3], name='img1')
    frame_img2 = tf.placeholder(tf.float32, [1, None, None, 3], name='img2')
    flow_est = pyramid_processing(frame_img1, frame_img2, train=False, trainable=False, regularizer=None, is_scale=True)
    flow_est_color = flow_to_color(flow_est['full_res'], mask=None, max_flow=256)

    opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(var_list=restore_vars)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=opts))

    print("- Initialize architecture")
    sess.run(tf.global_variables_initializer())

    print("- Restore weights")
    saver.restore(sess, restore_model)
    print("--- DONE ---")
    return sess, flow_est, flow_est_color, scale

# Run the Flow estimation model based on a pair of frames
def run_fe_model(fe_model, pair):
    sess, flow_est, flow_est_color, scale = fe_model

    # Load frame 1 and normalize it
    img1 = tf.image.decode_png(tf.read_file(pair.get_frames()[0].get_image_path()), channels=3)
    img1 = tf.cast(img1, tf.float32)
    img1 /= 255
    img1 = tf.expand_dims(img1, axis=0).eval(session=sess)

    # Load frame 2 and normalize it
    img2 = tf.image.decode_png(tf.read_file(pair.get_frames()[1].get_image_path()), channels=3)
    img2 = tf.cast(img2, tf.float32)
    img2 /= 255
    img2 = tf.expand_dims(img2, axis=0).eval(session=sess)

    # Run the model and output both the raw output and the colored demo image.
    np_flow_est, np_flow_est_color = sess.run([flow_est, flow_est_color], feed_dict={'img1:0': img1, 'img2:0': img2})
    return np_flow_est, np_flow_est_color
