import numpy as np
import torch
import math
import os

import datasets

from CSRNet.model import CSRNet

from utils import *

import scipy.misc
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import zoom
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw

# import tensorflow as tf
# from DDFlow.network import pyramid_processing
# from DDFlow.flowlib import flow_to_color

import utils

import datetime
#
# from DRNet.model import drnet_2D
# from DRNet.ini_file_io import load_train_ini

# Intialize the regionwise LOI. Returns all the information required to execute the regionwise LOI optimally.
# The initializer generates all the reqions around the LOI and the masks for extracting the data from the CC and FE.
def init_regionwise_loi(point1, point2, img_width, img_height, line_width, cregions, shrink):
    # Rescale everything
    center = (img_width / 2, img_height / 2)
    rotate_angle = math.degrees(math.atan((point2[1] - point1[1]) / float(point2[0] - point1[0])))

    # Generate the original regions as well for later usage
    regions = select_regions(point1, point2,
                             width=line_width, regions=cregions, shrink=shrink)

    # Generate per region a mask which can be used to extract the crowd counting and flow estimation
    masks = ([], [])
    for i, small_regions in enumerate(regions):
        for o, region in enumerate(small_regions):
            mask = region_to_mask(region, rotate_angle, center, img_width=img_width, img_height=img_height)
            masks[i].append(mask)

    return regions, rotate_angle, center, masks


# Combine both the crowd counting and the flow estimation with a regionwise method
# Returns per region how many people are crossing the line from that side.
def regionwise_loi(loi_info, counting_result, flow_result):
    regions, rotate_angle, center, masks = loi_info

    sums = ([], [])

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
            threshold = 0.5
            towards_pixels = fe_part > threshold

            total_crowd = cc_part.sum()

            # Too remove some noise
            if towards_pixels.sum() == 0 or total_crowd < 0:
                sums[i].append(0.0)
                continue

            towards_avg = fe_part[towards_pixels].sum() / float(towards_pixels.sum()) # float(total_pixels)
            away_pixels = fe_part < -threshold
            # away_avg = fe_part[away_pixels].sum() / away_pixels.sum()

            pixel_percentage_towards = towards_pixels.sum() / float(away_pixels.sum()+towards_pixels.sum())
            crowd_towards = total_crowd * pixel_percentage_towards

            # Divide the average
            percentage_over = towards_avg / region[4]

            sums[i].append(crowd_towards * percentage_over)

    return sums


def regionwise_loi_v2(loi_info, counting_result, flow_result):
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
            threshold = 0.5
            towards_pixels = fe_part > threshold

            total_crowd = cc_part.sum()

            # Too remove some noise
            if towards_pixels.sum() == 0 or total_crowd < 0:
                sums[i].append(0.0)
                continue

            towards_avg = fe_part[towards_pixels].sum() / float(towards_pixels.sum()) # float(total_pixels)
            away_pixels = fe_part < -threshold
            # away_avg = fe_part[away_pixels].sum() / away_pixels.sum()

            pixel_percentage_towards = towards_pixels.sum() / float(away_pixels.sum()+towards_pixels.sum())
            crowd_towards = total_crowd * pixel_percentage_towards

            # Divide the average
            percentage_over = towards_avg / region[4]

            sums[i].append(crowd_towards * percentage_over)

    return sums


# Initialize the crowd counting model
def init_cc_model(weights_path = 'CSRNet/fudan_only_model_best.pth.tar', img_width=1920, img_height=1080):
    print("--- LOADING CSRNET ---")
    print("- Initialize architecture")
    cc_model = CSRNet(load_weights=True)
    print("- Architecture to GPU")
    cc_model = cc_model.cuda()
    print("- Loading weights")
    checkpoint = torch.load(weights_path)
    print("- Weights to GPU")
    cc_model.load_state_dict(checkpoint['state_dict'])
    print("--- DONE ---")
    return cc_model, img_width, img_height


# Run the crowd counting model on frame A
def run_cc_model(cc_info, img):
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
    ])

    cc_model, img_width, img_height = cc_info

    # Normalize the image
    img = transform(img).cuda()

    cc_output = cc_model(img.unsqueeze(0))

    # Back to numpy and resize to original size
    cc_output = cc_output.detach().cpu().data.numpy().squeeze()
    cc_output = zoom(cc_output, zoom=8.0) / 64.

    return cc_output



def init_drnet_model(restore_model='DRNet/outcome/pretrain_B_new/pretrain_B_new-8', img_width=1920, img_height=1080):
    print("--- LOADING DRNET ---")

    print("- Load architecture")
    ini_file = 'DRNet/ini/tr_param.ini'
    param_sets = load_train_ini(ini_file)
    param_set = param_sets[0]
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    model = drnet_2D(sess, param_set)

    print("- Initialize architecture")
    model.sess.run(tf.global_variables_initializer())

    print("- Restore weights")
    model.saver.restore(model.sess, restore_model)

    print("--- DONE ---")
    return model


def run_drnet_model(drnet_model, frame):
    # Load image
    img_data = np.array(Image.open(frame.get_image_path()))  # .convert('L'))

    # Preprocess image
    if (len(img_data.shape) < 3):
        img_data = imArr[:, :, np.newaxis]
        img_data = np.tile(img_data, (1, 1, 3))

    img_data = img_data.astype('float32')
    img_data = img_data / 255.0
    w, h, c = img_data.shape
    w = int(w / 4) * 4
    h = int(h / 4) * 4
    img_data = resize(img_data, (w, h, c), preserve_range=True)
    img_data = img_data.reshape(1, w, h, c)

    predicted_label = self.sess.run(drnet_model.pred_prob, feed_dict={drnet_model.input_Img: img_data})

    predicted_label /= 100.0
    predicted_label = np.squeeze(predicted_label)

    return predicted_label

from DDFlow_pytorch.model import PWCNet

def init_fe_model_v2(restore_model='DDFlow_pytorch/weights/20200618_104414_tub_d1/model_150000.pt', img_width=1920, img_height=1080, display=False):
    if display:
        print("--- LOADING DDFLOW ---")
    if display:
        print("- Initialize architecture")
    net = PWCNet()
    if display:
        print("- Restore weights")
    net.load_state_dict(torch.load(restore_model))
    if display:
        print("- To GPU")
    if display:
        print("--- DONE ---")
    net = net.cuda()
    net.eval()
    return net, img_width, img_height

def run_fe_model_v2(fe_model, pair):
    net, img_width, img_height = fe_model

    frame1 = torch.FloatTensor(np.ascontiguousarray(
        np.array(Image.open(pair.get_frames(0).get_image_path()))[:, :, ::-1].transpose(2, 0, 1).astype(
            np.float32) * (1.0 / 255.0)))
    frame2 = torch.FloatTensor(np.ascontiguousarray(
        np.array(Image.open(pair.get_frames(1).get_image_path()))[:, :, ::-1].transpose(2, 0, 1).astype(
            np.float32) * (1.0 / 255.0)))

    frame1 = frame1.cuda()
    frame2 = frame2.cuda()
    frame1 = frame1.unsqueeze(0)
    frame2 = frame2.unsqueeze(0)

    flow = net.single_forward(frame1, frame2)
    flow = flow.squeeze().permute(1,2,0)

    return flow.detach().cpu().data.numpy()