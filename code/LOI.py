import numpy as np
import torch
import math

import datasets
import LOI

import scipy.misc
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import zoom


from CSRNet.model import CSRNet
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw

import tensorflow as tf
from DDFlow.network import pyramid_processing
from DDFlow.flowlib import flow_to_color

def pixelwise_loi(counting_result, flow_result):
    return

def region_to_mask(region, img_width=1920, img_height=1080):
    # full_size = sqrt(pow(img_height) + pow(img_width))

    p1 = region[0]
    p2 = region[2]
    mask = np.zeros((img_width, img_height))
    mask[min(p1[0], p2[0]):max(p1[0], p2[0]), min(p1[1], p2[1]):max(p1[1], p2[1])] = 1
    return mask

def init_regionwise_loi(point1, point2, img_width=1920, img_height=1080, width=50, cregions=5, shrink=0.90):
    center = (img_width / 2, img_height / 2)
    rotate_angle = math.degrees(math.atan((point2[1] - point1[1]) / float(point2[0] - point1[0])))

    regions = select_regions(rotate_point(point1, -rotate_angle, center), rotate_point(point2, -rotate_angle, center),
                             width=width, regions=cregions, shrink=shrink)

    masks = ([], [])
    for i, small_regions in enumerate(regions):
        for o, region in enumerate(small_regions):
            masks[i].append(region_to_mask(region))

    return regions, rotate_angle, center, masks

def regionwise_loi(counting_result, flow_result, loi_info):
    regions, rotate_angle, center, masks = loi_info

    sums = ([], [])

    counting_result = rotate(counting_result, rotate_angle, reshape=False)
    flow_result = np.squeeze(rotate(flow_result['full_res'], rotate_angle, reshape=False))

    for i, small_regions in enumerate(regions):
        for o, region in enumerate(small_regions):
            cc_part = (masks[i][o].transpose()+ 0.5) * counting_result
            img = Image.fromarray(cc_part * 255.0 / max(0.001, cc_part.max()))
            img = img.convert("L")
            img.save('results/mask.png')

            fe_part = np.expand_dims(masks[i][o].transpose(), axis=2) * flow_result

            fe_l = fe_part[:, :, 0]

            sums[i].append((cc_part.sum(), fe_l[fe_l > 0].sum(), fe_l[fe_l <= 0].sum()))

            # if i == 0:
            #     over = fe_l[fe_l > 0].sum()
            # elif i == 1:
            #     over = fe_l[fe_l <= 0].sum()

    return sums


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


def select_regions(dot1, dot2, width=50, regions=5, shrink=0.90):
    region_lines = []
    line_points = np.linspace(np.array(dot1), np.array(dot2), num=regions+1).astype(int)
    for i, point in enumerate(line_points):
        if i + 1 >= len(line_points):
            break

        point2 = line_points[i + 1]
        region_lines.append((tuple(list(point)),  tuple(list(point2))))

    regions = ([], [])

    region_lines.reverse()

    for point1, point2 in region_lines:
        part_line_length = math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))
        point_diff = (
            - (point1[1] - point2[1]) / float(part_line_length) *  float(width),
            (point1[0] - point2[0]) / float(part_line_length) * float(width)
        )

        width *= shrink

        regions[0].append([
            point1,
            point2,
            (int(point2[0] + point_diff[0]), int(point2[1] + point_diff[1])),
            (int(point1[0] + point_diff[0]), int(point1[1] + point_diff[1]))
        ])

        regions[1].append([
            point1,
            point2,
            (int(point2[0] - point_diff[0]), int(point2[1] - point_diff[1])),
            (int(point1[0] - point_diff[0]), int(point1[1] - point_diff[1]))
        ])

    regions[0].reverse()
    regions[1].reverse()

    return regions


def image_add_region_lines(image, dot1, dot2, loi_output=None, width=50, nregions=5, shrink=0.90):
    regions = select_regions(dot1, dot2, width=width, regions=nregions, shrink=shrink)
    draw = ImageDraw.Draw(image)

    for i, small_regions in enumerate(regions):
        for o, region in enumerate(small_regions):

            if i == 0:
                outline_c = 100
            else:
                outline_c = 300

            draw.polygon(region, outline=outline_c)

            if loi_output:
                msg = str(loi_output[i][o])
                w, h = draw.textsize(str(msg))
                center = (
                    (region[0][0] + region[2][0] - w) / 2,
                    (region[0][1] + region[2][1] - h) / 2
                )
                draw.text(center, msg, fill="white")

    draw.line((dot1[0], dot1[1], dot2[0], dot2[1]), fill=200, width=10)


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
    img = 255.0 * F.to_tensor(Image.open(frame.get_image_path()).convert('RGB'))
    img[0, :, :] = img[0, :, :] - 92.8207477031
    img[1, :, :] = img[1, :, :] - 95.2757037428
    img[2, :, :] = img[2, :, :] - 104.877445883
    img = img.cuda()
    cc_output = cc_model(img.unsqueeze(0))
    cc_output = cc_output.detach().cpu().data.numpy().squeeze()

    return zoom(cc_output, zoom=8.0) / 64.


# Initialize the flow estimation model
def init_fe_model(restore_model = 'DDFlow/Fudan/checkpoints/distillation_census_prekitty2/model-70000'):
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


def run_fe_model(fe_model, pair):
    sess, flow_est, flow_est_color = fe_model

    img1 = tf.image.decode_png(tf.read_file(pair.get_frames()[0].get_image_path()), channels=3)
    img1 = tf.cast(img1, tf.float32)
    img1 /= 255
    img1 = tf.expand_dims(img1, axis=0).eval(session=sess)
    #img1 = tf.identity(img1, name="img1")

    img2 = tf.image.decode_png(tf.read_file(pair.get_frames()[1].get_image_path()), channels=3)
    img2 = tf.cast(img2, tf.float32)
    img2 /= 255
    img2 = tf.expand_dims(img2, axis=0).eval(session=sess)

    np_flow_est, np_flow_est_color = sess.run([flow_est, flow_est_color], feed_dict={'img1:0': img1, 'img2:0': img2})
    # img1, img2, np_flow_est, np_flow_est_color = sess.run([img1, img2, flow_est, flow_est_color])

    return np_flow_est, np_flow_est_color
