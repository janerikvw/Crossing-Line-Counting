import numpy as np
import torch
import math

import datasets

import scipy.misc
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import zoom

from CSRNet.model import CSRNet
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw

import tensorflow as tf
from DDFlow.network import pyramid_processing
from DDFlow.flowlib import flow_to_color

# Give a region and turn it in a mask to extract the region information
def region_to_mask(region, img_width=1920, img_height=1080):
    full_size = int(math.sqrt(math.pow(img_height, 2) + math.pow(img_width, 2)))

    p1 = region[0]
    p2 = region[2]
    # Full size is the maximum size an image can get. (When an image is rotated 45 degrees it get's this size
    mask = np.zeros((full_size, full_size))

    fw = int((full_size - img_width)/2)
    fh = int((full_size - img_height)/2)

    # Add the region mask to the empty mask
    mask[fw + min(p1[0], p2[0]):fw + max(p1[0], p2[0]), fh + min(p1[1], p2[1]):fh + max(p1[1], p2[1])] = 1
    return mask


# Rotate an individual point with a certain angle and a given centre
# In case of the LOI the centre is always the middle of the image
# A standard Linear Algebra trick
def rotate_point(point, angle, center):
    r_angle = math.radians(angle)
    r00 = math.cos(r_angle)
    r01 = -math.sin(r_angle)
    r10 = math.sin(r_angle)
    r11 = math.cos(r_angle)

    out = (
        r00 * point[0] + r01 * point[1] + center[0] - r00 * center[0] - r01 * center[1],
        r10 * point[0] + r11 * point[1] + center[1] - r10 * center[0] - r11 * center[1]
    )
    return out

# Generate all the regions around the LOI (given by dot1 and dot2).
# A region is an array with all the cornerpoints and the with of the region
def select_regions(dot1, dot2, width=50, regions=5, shrink=0.90):
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
def image_add_region_lines(image, dot1, dot2, loi_output=None, width=50, nregions=5, shrink=0.90):
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
def add_total_information(image, loi_output, totals):
    draw = ImageDraw.Draw(image)

    to_right = sum(loi_output[1])
    to_left = sum(loi_output[0])

    msg = 'Current: ({:.3f}, {:.3f}), Total: ({:.3f}, {:.3f})'.format(to_right, to_left, totals[1], totals[0])
    w, h = draw.textsize(msg)

    width, height = image.size
    draw.rectangle([
        20, height - h - 20,
            20 + w, height - 20
    ], fill="black")
    draw.text((20, height - h - 20), msg, fill="white")
    