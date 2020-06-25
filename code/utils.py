import numpy as np
import torch
import math
import os
from scipy import misc, io

import datasets

import scipy.misc
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import zoom

from CSRNet.model import CSRNet
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw

from scipy.ndimage import rotate

# Give a region and turn it in a mask to extract the region information
def region_to_mask(region, rotate_angle, center, img_width, img_height):
    full_size = int(math.sqrt(math.pow(img_height, 2) + math.pow(img_width, 2)))

    p1 = rotate_point(region[0], -rotate_angle, center)
    p2 = rotate_point(region[2], -rotate_angle, center)

    # Full size is the maximum size an image can get. (When an image is rotated 45 degrees it get's this size
    mask = np.zeros((full_size, full_size))

    fw = int((full_size - img_width)/2)
    fh = int((full_size - img_height)/2)

    # Add the region mask to the empty mask
    mask[fw + min(p1[0], p2[0]):fw + max(p1[0], p2[0]), fh + min(p1[1], p2[1]):fh + max(p1[1], p2[1])] = 1

    mask = rotate(mask, rotate_angle, reshape=False)
    mask = mask[fw:-fw, fh:-fh]
    mask = mask.transpose()
    return mask


# Rotate an individual point with a certain angle and a given centre
# In case of the LOI the centre is always the middle of the image
# A standard Linear Algebra trick
def rotate_point(point, angle, center, to_int=True):
    r_angle = math.radians(angle)
    r00 = math.cos(r_angle)
    r01 = -math.sin(r_angle)
    r10 = math.sin(r_angle)
    r11 = math.cos(r_angle)

    out = (
        r00 * point[0] + r01 * point[1] + center[0] - r00 * center[0] - r01 * center[1],
        r10 * point[0] + r11 * point[1] + center[1] - r10 * center[0] - r11 * center[1]
    )

    if to_int:
        out = int(out[0]), int(out[1])

    return out


# Generate all the regions around the LOI (given by dot1 and dot2).
# A region is an array with all the cornerpoints and the with of the region
def select_regions(dot1, dot2, width, regions, shrink):
    # Seperate the line into several parts with given start and end point.
    # Provide the corner points of the regions that lie on the LOI itself.
    region_lines = []
    line_points = np.linspace(np.array(dot1), np.array(dot2), num=regions+1).astype(int)
    for i, point in enumerate(line_points):
        if i + 1 >= len(line_points):
            break

        point2 = line_points[i + 1]
        region_lines.append((tuple(list(point)),  tuple(list(point2))))

    region_lines.reverse()
    regions = ([], [])
    for point1, point2 in region_lines:

        # The difference which we can add to the region lines corners
        # to come to the corners on the other end of the region
        part_line_length = math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))
        point_diff = (
            - (point1[1] - point2[1]) / float(part_line_length) * float(width),
            (point1[0] - point2[0]) / float(part_line_length) * float(width)
        )

        # Both add and substract the difference so we get the regions on both sides of the LOI.
        regions[0].append([
            point1,
            point2,
            (int(point2[0] + point_diff[0]), int(point2[1] + point_diff[1])),
            (int(point1[0] + point_diff[0]), int(point1[1] + point_diff[1])),
            width
        ])

        regions[1].append([
            point1,
            point2,
            (int(point2[0] - point_diff[0]), int(point2[1] - point_diff[1])),
            (int(point1[0] - point_diff[0]), int(point1[1] - point_diff[1])),
            width
        ])

        # Shrink the width for perspective
        width *= shrink

    regions[0].reverse()
    regions[1].reverse()

    return regions


# Add the regions and LOI to an PIL image. The information of the LOI can be added for each region as well.
def image_add_region_lines(image, dot1, dot2, width, nregions, shrink, loi_output=None):
    regions = select_regions(dot1, dot2, width=width, regions=nregions, shrink=shrink)
    draw = ImageDraw.Draw(image)

    for i, small_regions in enumerate(regions):
        for o, region in enumerate(small_regions):

            if i == 0:
                outline_c = 100
            else:
                outline_c = 300

            draw.polygon(region[0:4], outline=outline_c)

            if loi_output:
                msg = loi_output[i][o]
                msg = '{:.3f}'.format(msg)
                w, h = draw.textsize(msg)
                center = (
                    (region[0][0] + region[2][0] - w) / 2,
                    (region[0][1] + region[2][1] - h) / 2
                )
                draw.text(center, msg, fill="white")

    # Add the action LOI line in the middle
    draw.line((dot1[0], dot1[1], dot2[0], dot2[1]), fill=200, width=10)


# Add the current/total information at the bottom of an image
def add_total_information(image, loi_output, totals, crosses):
    draw = ImageDraw.Draw(image)

    to_right = sum(loi_output[1])
    to_left = sum(loi_output[0])

    msg = '{:.2f}, {:.2f} ({:.2f}, {:.2f}) - ({}, {})'.format(to_right, to_left, totals[1], totals[0], len(crosses[0]), len(crosses[1]))
    w, h = draw.textsize(msg)

    width, height = image.size
    draw.rectangle([
        20, height - h - 20,
            20 + w, height - 20
    ], fill="black")
    draw.text((20, height - h - 20), msg, fill="white")

def white_to_transparency_gradient(img):
    x = np.asarray(img.convert('RGBA')).copy()

    x[:, :, 3] = (255 - np.power(x[:, :, :3].mean(axis=2)/255., 3)*255/3).astype(np.uint8)

    return Image.fromarray(x)

def scale_point(point1, scale):
    point1 = (point1[0] * scale, point1[1] * scale)
    return point1

# Scale all parameters correctly when it changes from the original size
def scale_frame(frame_width, frame_height, scale):
    frame_width = int(frame_width * scale)
    frame_height = int(frame_height * scale)
    return frame_width, frame_height

def store_processed_images(result_dir, sample_name, print_i, frame1_img, cc_output, fe_output_color, point1, point2, line_width, nregions, orientation_shrink, loi_output, totals, crosses):
    # Load original image or demo
    total_image = frame1_img.copy().convert("L").convert("RGBA")

    if sample_name:
        dump_dir = os.path.join(result_dir, sample_name)
    else:
        dump_dir = result_dir

    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)

    # Load and save original CC model
    cc_img = Image.fromarray(cc_output * 255.0 / cc_output.max())
    cc_img = cc_img.convert("L")
    cc_img.save(os.path.join(dump_dir, 'crowd_{}.png').format(print_i))

    # Transform CC model and merge with original image for demo
    cc_img = np.zeros((cc_output.shape[0], cc_output.shape[1], 4))
    cc_img = cc_img.astype(np.uint8)
    cc_img[:, :, 3] = 255 - (cc_output * 255.0 / cc_output.max())
    cc_img[cc_img > 2] = cc_img[cc_img > 2] * 0.4
    cc_img = Image.fromarray(cc_img, 'RGBA')
    total_image = Image.alpha_composite(total_image, cc_img)

    # Save original flow image
    misc.imsave(os.path.join(dump_dir, 'flow_{}.png'.format(print_i)), fe_output_color[0])

    # Transform flow and merge with original image
    flow_img = fe_output_color[0] * 255.0
    flow_img = flow_img.astype(np.uint8)
    flow_img = Image.fromarray(flow_img, 'RGB')
    flow_img = white_to_transparency_gradient(flow_img)
    total_image = Image.alpha_composite(total_image, flow_img)

    # Save merged image
    total_image.save(os.path.join(dump_dir, 'combined_{}.png'.format(print_i)))
    img = total_image.convert('RGB')
    # Generate the demo output for clear view on what happens
    image_add_region_lines(img, point1, point2, width=line_width, nregions=nregions, shrink=orientation_shrink, loi_output=loi_output)

    # Crop and add information at the bottom of the screen
    add_total_information(img, loi_output, totals, crosses)

    img.save(os.path.join(dump_dir, 'line_{}.png'.format(print_i)))

import datetime
class sTimer():
    def __init__(self, name):
        self.start = datetime.datetime.now()
        self.name = name

    def show(self, printer=True):
        ms = int((datetime.datetime.now() - self.start).total_seconds() * 1000)
        if printer:
            print("{}: {}ms".format(self.name, ms))

        return ms