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

# Intialize the regionwise LOI. Returns all the information required to execute the regionwise LOI optimally.
# The initializer generates all the reqions around the LOI and the masks for extracting the data from the CC and FE.
def init_regionwise_loi(point1, point2, img_width=1920, img_height=1080, width=50, cregions=5, shrink=0.90):
    center = (img_width / 2, img_height / 2)
    rotate_angle = math.degrees(math.atan((point2[1] - point1[1]) / float(point2[0] - point1[0])))

    # Select regions on a rotated image, so all the regions can be used to create simple masks
    regions = select_regions(rotate_point(point1, -rotate_angle, center), rotate_point(point2, -rotate_angle, center),
                             width=width, regions=cregions, shrink=shrink)

    # Generate the original regions as well for later usage
    regions_orig = select_regions(point1, point2,
                             width=width, regions=cregions, shrink=shrink)

    # Generate per region a mask which can be used to extract the crowd counting and flow estimation
    masks = ([], [])
    for i, small_regions in enumerate(regions):
        for o, region in enumerate(small_regions):
            masks[i].append(region_to_mask(region))

    return regions, rotate_angle, center, masks, regions_orig

# Combine both the crowd counting and the flow estimation with a regionwise method
# Returns per region how many people are crossing the line from that side.
def regionwise_loi(loi_info, counting_result, flow_result):
    regions, rotate_angle, center, masks, regions_orig = loi_info

    sums = ([], [])

    # Get all the sizes required to correctly extract the regions
    ow = counting_result.shape[0]
    oh = counting_result.shape[1]
    full_size = int(math.sqrt(math.pow(oh, 2) + math.pow(ow, 2)))
    f_counting_result = np.zeros((full_size, full_size))
    f_flow_result = np.zeros((full_size, full_size, 2))
    fw = int((full_size - counting_result.shape[0]) / 2)
    fh = int((full_size - counting_result.shape[1]) / 2)

    # Put the earlier results in bigger array to prevent rotation out of the image during rotation
    f_counting_result[fw:fw+ow, fh:oh+fh] = counting_result
    f_flow_result[fw:fw+ow, fh:oh+fh] = flow_result['full_res']

    # Rotate the images.
    counting_result = rotate(f_counting_result, rotate_angle, reshape=False)
    flow_result = np.squeeze(rotate(f_flow_result, rotate_angle, reshape=False))

    # @TODO: small_regions to something more correct
    for i, small_regions in enumerate(regions):
        for o, region in enumerate(small_regions):
            # Get the CC region
            cc_part = (masks[i][o].transpose()) * counting_result

            # Project the flow estimation output on a line perpendicular to the LOI,
            # so we can calculate if the people are approach/leaving the LOI.
            region_orig = regions_orig[i][o]
            direction = np.array([region_orig[1][0] - region_orig[2][0], region_orig[1][1] - region_orig[2][1]]).astype(
                np.float32)
            direction = direction / np.linalg.norm(direction)
            perp = np.sum(np.multiply(flow_result, direction), axis=2)
            # Get the FE region
            fe_part = masks[i][o].transpose() * perp

            # Get all the movement towards the line
            total_pixels = masks[i][o].sum()

            threshold = 1
            towards_pixels = fe_part > threshold

            total_crowd = cc_part.sum()

            if towards_pixels.sum() == 0 or total_crowd < 0:
                sums[i].append(0.0)
                continue

            towards_avg = fe_part[towards_pixels].sum() / float(total_pixels) # float(towards_pixels.sum())
            away_pixels = fe_part < -threshold
            # away_avg = fe_part[away_pixels].sum() / away_pixels.sum()

            # print('towards_pixels', towards_pixels.sum())
            # print('towards avg', towards_avg)
            # print('away_pixels', away_pixels.sum())

            pixel_percentage_towards = towards_pixels.sum() / float(away_pixels.sum()+towards_pixels.sum())
            crowd_towards = total_crowd * pixel_percentage_towards

            # print('pixel_percentage_towards', pixel_percentage_towards)
            # print('crowd_towards', crowd_towards)

            # Divide the average
            percentage_over = towards_avg / region[4]
            # print('percentage_over', percentage_over)
            # print('End', crowd_towards * percentage_over)

            sums[i].append(crowd_towards * percentage_over)


    return sums


# Initialize the crowd counting model
def init_cc_model(weights_path = 'CSRNet/fudan_only_model_best.pth.tar'):
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
    return cc_model


# Run the crowd counting model on frame A
def run_cc_model(cc_model, frame):
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
def init_fe_model(restore_model='DDFlow/Fudan/checkpoints/distillation_census_prekitty2/model-70000'):
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
    return sess, flow_est, flow_est_color

# Run the Flow estimation model based on a pair of frames
def run_fe_model(fe_model, pair):
    sess, flow_est, flow_est_color = fe_model

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
