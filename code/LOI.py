import numpy as np
import torch
import math
import os

import datasets

from CSRNet.model import CSRNet
from DDFlow_pytorch.model import PWCNet

from utils import *

import scipy.misc
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import zoom
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw

from torchvision import datasets, transforms

import utils

import datetime

# Intialize the regionwise LOI. Returns all the information required to execute the regionwise LOI optimally.
# The initializer generates all the reqions around the LOI and the masks for extracting the data from the CC and FE.
def init_regionwise_loi(point1, point2, img_width, img_height, ped_size,  width_peds, height_peds, select_type='V2', crop_processing=False):
    # Rescale everything
    center = (img_width / 2, img_height / 2)
    rotate_angle = math.degrees(math.atan((point2[1] - point1[1]) / float(point2[0] - point1[0])))

    # Generate the original regions as well for later usage
    if select_type == 'V1':
        regions = select_regions_v1(point1, point2, width=35*height_peds, regions=5, shrink=1.0)
    else:
        regions = select_regions(point1, point2, ped_size=ped_size, width_peds=width_peds, height_peds=height_peds)

    # Generate per region a mask which can be used to extract the crowd counting and flow estimation
    masks = ([], [])
    for i, small_regions in enumerate(regions):
        for o, region in enumerate(small_regions):
            mask = region_to_mask(region, rotate_angle, img_width=img_width, img_height=img_height)
            masks[i].append(mask)

    return regions, rotate_angle, center, masks, {'crop': crop_processing, 'width': img_width, 'height': img_height}


def select_line_outer_points(regions, crop_distance, width, height):
    arr = np.asarray(np.array(regions)[:, :, :-1].flatten().tolist())
    min_x, max_x, min_y, max_y = np.min(arr[:, 0]), np.max(arr[:, 0]), \
                                 np.min(arr[:, 1]), np.max(arr[:, 1])

    if crop_distance is True:
        crop_distance = 0

    min_x = max(min_x-crop_distance, 0)
    max_x = min(max_x + crop_distance, width)
    min_y = max(min_y - crop_distance, 0)
    max_y = min(max_y + crop_distance, height)

    return min_x, max_x, min_y, max_y

def preprocess_image_loi(loi, image):
    regions, rotate_angle, center, masks, crop = loi
    if crop['crop'] is False:
        return image

    min_x, max_x, min_y, max_y = select_line_outer_points(regions, crop['crop'], crop['width'], crop['height'])
    image = image.crop((min_x, min_y, max_x, max_y))

    image.save(os.path.join('orig_crop.png'))

    return image






# Combine both the crowd counting and the flow estimation with a regionwise method
# Returns per region how many people are crossing the line from that side.
def regionwise_loi(loi_info, counting_result, flow_result):
    regions, rotate_angle, center, masks, crop = loi_info

    sums = ([], [])

    # @TODO: small_regions to something more correct
    for i, side_regions in enumerate(regions):
        for o, region in enumerate(side_regions):
            mask = masks[i][o]

            # Get which part of the mask contains the actual mask
            # This massively improves the speed of the model
            points = np.array(region[0:4])

            # Get the crop for the regions
            min_x, max_x, min_y, max_y = np.min(points[:, 0]), np.max(points[:, 0]),\
                                         np.min(points[:, 1]), np.max(points[:, 1])

            # Adjust the crop if we crop the prediction image as well
            if crop['crop'] is False:
                lc_min_x, lc_max_x, lc_min_y, lc_max_y = min_x, max_x, min_y, max_y
            else:
                adjust_x, _, adjust_y, _ = select_line_outer_points(regions, crop['crop'], crop['width'], crop['height'])
                lc_min_x, lc_max_x, lc_min_y, lc_max_y = min_x-adjust_x, max_x-adjust_x, min_y-adjust_y, max_y-adjust_y

            cropped_mask = mask[min_y:max_y, min_x:max_x]

            # Use cropped mask on crowd counting result
            cc_part = cropped_mask * counting_result[lc_min_y:lc_max_y, lc_min_x:lc_max_x]

            # Project the flow estimation output on a line perpendicular to the LOI,
            # so we can calculate if the people are approach/leaving the LOI.
            direction = np.array([region[1][0] - region[2][0], region[1][1] - region[2][1]]).astype(
                np.float32)
            direction = direction / np.linalg.norm(direction)

            # Crop so only cropped area gets projected
            part_flow_result = flow_result[lc_min_y:lc_max_y, lc_min_x:lc_max_x]
            perp = np.sum(np.multiply(part_flow_result, direction), axis=2)
            fe_part = cropped_mask * perp

            # Get all the movement towards the line
            # total_pixels = masks[i][o].sum()
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

# Combine both the crowd counting and the flow estimation with a regionwise method
# Returns per region how many people are crossing the line from that side.
def pixelwise_loi(loi_info, counting_result, flow_result):
    regions, rotate_angle, center, masks, crop = loi_info

    sums = ([], [])

    # @TODO: small_regions to something more correct
    for i, side_regions in enumerate(regions):
        for o, region in enumerate(side_regions):
            mask = masks[i][o]

            # Get which part of the mask contains the actual mask
            # This massively improves the speed of the model
            points = np.array(region[0:4])

            # Get the crop for the regions
            min_x, max_x, min_y, max_y = np.min(points[:, 0]), np.max(points[:, 0]),\
                                         np.min(points[:, 1]), np.max(points[:, 1])

            # Adjust the crop if we crop the prediction image as well
            if crop['crop'] is False:
                lc_min_x, lc_max_x, lc_min_y, lc_max_y = min_x, max_x, min_y, max_y
            else:
                adjust_x, _, adjust_y, _ = select_line_outer_points(regions, crop['crop'], crop['width'], crop['height'])
                lc_min_x, lc_max_x, lc_min_y, lc_max_y = min_x-adjust_x, max_x-adjust_x, min_y-adjust_y, max_y-adjust_y

            cropped_mask = mask[min_y:max_y, min_x:max_x]

            # Use cropped mask on crowd counting result
            cc_part = cropped_mask * counting_result[lc_min_y:lc_max_y, lc_min_x:lc_max_x]

            # Project the flow estimation output on a line perpendicular to the LOI,
            # so we can calculate if the people are approach/leaving the LOI.
            direction = np.array([region[1][0] - region[2][0], region[1][1] - region[2][1]]).astype(
                np.float32)
            direction = direction / np.linalg.norm(direction)

            # Crop so only cropped area gets projected
            part_flow_result = flow_result[lc_min_y:lc_max_y, lc_min_x:lc_max_x]

            perp = np.sum(np.multiply(part_flow_result, direction), axis=2)
            fe_part = cropped_mask * perp

            # Get all the movement towards the line
            # total_pixels = masks[i][o].sum()
            threshold = 0.5
            towards_pixels = fe_part > threshold

            # Too remove some noise
            if towards_pixels.sum() == 0 or cc_part.sum() < 0:
                sums[i].append(0.0)
                continue

            sums[i].append(np.multiply(fe_part[towards_pixels], cc_part[towards_pixels]).sum() / region[4])

    return sums


# Initialize the crowd counting model
def init_cc_model(weights_path, img_width=1920, img_height=1080):
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
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
    ])

    cc_model, img_width, img_height = cc_info

    # Normalize the image
    img = transform(img).cuda()

    cc_output = cc_model(img.unsqueeze(0))

    factor = (img.shape[1] * img.shape[2]) / (cc_output.shape[2] * cc_output.shape[3])
    #print("CC Output diff")
    #print(cc_output.sum(), cc_output.shape)

    cc_output = torch.nn.functional.interpolate(input=cc_output,
                              size=(img.shape[1], img.shape[2]),
                              mode='bilinear',
                              align_corners=False) / factor

    #print(cc_output.sum(), cc_output.shape)

    # Back to numpy and resize to original size
    cc_output = cc_output.detach().cpu().data.numpy().squeeze()
    # cc_output = zoom(cc_output, zoom=8.0) / 64.

    return cc_output

def init_fe_model(weights_path, img_width=1920, img_height=1080, display=False):
    if display:
        print("--- LOADING DDFLOW ---")
    if display:
        print("- Initialize architecture")
    net = PWCNet()
    if display:
        print("- Restore weights")
    net.load_state_dict(torch.load(weights_path))
    if display:
        print("- To GPU")
    net = net.cuda()
    if display:
        print("--- DONE ---")
    net.eval()
    return net, img_width, img_height

def run_fe_model(fe_model, frame1_pil, frame2_pil):
    net, img_width, img_height = fe_model

    frame1 = torch.FloatTensor(
        np.array(frame1_pil)[:, :, ::-1].transpose(2, 0, 1).astype(
            np.float32) * (1.0 / 255.0))
    frame2 = torch.FloatTensor(
        np.array(frame2_pil)[:, :, ::-1].transpose(2, 0, 1).astype(
            np.float32) * (1.0 / 255.0))
    frame1 = frame1.cuda()
    frame2 = frame2.cuda()
    frame1 = frame1.unsqueeze(0)
    frame2 = frame2.unsqueeze(0)

    flow = net.single_forward(frame1, frame2)
    flow = flow.squeeze().permute(1,2,0)
    ret = flow.detach().cpu().data.numpy()

    return ret


#from full_on_pwc.models import ModelV2 as MergedModel
import torch.nn.functional as FU

def init_full_model(weights_path, display=False):
    if display:
        print("--- LOADING MERGED MODEL ---")
    if display:
        print("- Initialize architecture")
    net = MergedModel()
    if display:
        print("- Restore weights")
    net.load_state_dict(torch.load(weights_path))
    if display:
        print("- To GPU")
    net = net.cuda()
    if display:
        print("--- DONE ---")
    net.eval()
    return net


def run_full_model(full_model, frame1_pil, frame2_pil):
    # Load the image and normalize
    frame1 = torch.FloatTensor(np.ascontiguousarray(
        np.array(frame1_pil)[:, :, ::-1].transpose(2, 0, 1).astype(
            np.float32) * (1.0 / 255.0))).cuda()
    frame2 = torch.FloatTensor(np.ascontiguousarray(
        np.array(frame2_pil)[:, :, ::-1].transpose(2, 0, 1).astype(
            np.float32) * (1.0 / 255.0))).cuda()
    frame1.unsqueeze_(0)
    frame2.unsqueeze_(0)

    flow, _, density = full_model(frame1, frame2)

    factor = (frame1.shape[2] * frame1.shape[3]) / (density.shape[2] * density.shape[3])

    print(frame1)

    density = FU.interpolate(input=density,
                              size=(frame1.shape[2], frame1.shape[3]),
                              mode='bicubic', align_corners=False) / factor
    flow = flow.squeeze().permute(1, 2, 0)
    flow = flow.detach().cpu().data.numpy()

    density = density.detach().cpu().data.numpy().squeeze()

    return flow, density
