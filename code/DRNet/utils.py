import math
import numpy as np
import scipy.io as sio
from skimage.transform import rescale, resize
import copy
from random import *
from PIL import Image
from scipy.ndimage.filters import gaussian_filter, uniform_filter
from scipy.spatial import KDTree

import tensorflow as tf


def SaveDmap(predicted_label, labeling_path):
    # sio.savemat(labeling_path+'.mat', {'dmap':predicted_label})

    predicted_label = (predicted_label - np.min(predicted_label)) / (np.max(predicted_label) - np.min(predicted_label))
    img = Image.fromarray(np.array(predicted_label * 255.0).astype('uint8'))
    img.save(labeling_path + '.jpg')


def ReadImage(imPath):
    """
    Read gray images.
    """
    imArr = np.array(Image.open(imPath))  # .convert('L'))

    if (len(imArr.shape) < 3):
        imArr = imArr[:, :, np.newaxis]
        imArr = np.tile(imArr, (1, 1, 3))

    return imArr


def ResizeDmap(densitymap, scale=1.0):
    b, w, h = densitymap.shape
    rescale_densitymap = np.zeros([b, int(w * scale), int(h * scale)]).astype('float32')
    for i in xrange(b):
        dmap_sum = densitymap[i, :, :].sum()
        rescale_densitymap[i, :, :] = rescale(densitymap[i, :, :], scale, preserve_range=True)  #
        res_sum = rescale_densitymap[i, :, :].sum()
        if res_sum != 0:
            rescale_densitymap[i, :, :] = rescale_densitymap[i, :, :] * (dmap_sum / res_sum)
        # densitymap = densitymap.reshape(1,densitymap.shape[0],densitymap.shape[1])
    return rescale_densitymap


def ReadMap(mapPath, name):
    """
    Load the density map from matfile.
    """
    map_data = sio.loadmat(mapPath)
    return map_data[name]


def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))
    leafsize = 2048
    # build kdtree
    tree = KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += gaussian_filter(pt2d, sigma, mode='constant')
    return density


def gaussian_filter_density_fix(gt, sigma=10):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        density += gaussian_filter(pt2d, sigma, mode='constant')
    return density


def gaussian_filter_density_uniform(gt, sigma=10.0, d=4):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))
    u_size = ((12.0 / d) * sigma ** 2) ** 0.5
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        for j in xrange(0, d):
            pt2d = uniform_filter(pt2d, u_size, mode='constant')

        density += pt2d
    return density


def get_dmap(img_path, dmap_path):
    img_data = np.array(Image.open(img_path))
    img_width, img_height, im_channel = img_data.shape
    ann_data = sio.loadmat(dmap_path)['image_info']
    ann_data = ann_data[0, 0][0, 0][0].astype(np.float32).round().astype(int)
    ann_data[:, 0] = ann_data[:, 0].clip(0, img_height - 1)
    ann_data[:, 1] = ann_data[:, 1].clip(0, img_width - 1)

    dmap_data = np.zeros((img_width, img_height), dtype=np.float32)
    dmap_data[(ann_data[:, 1], ann_data[:, 0])] = 1

    return dmap_data


def load_data_pairs(img_list, dmap_list):
    """load all volume pairs"""
    img_clec = []
    dmap_clec = []

    for k in range(0, len(img_list)):
        img_data = ReadImage(img_list[k])
        img_width, img_height, im_channel = img_data.shape

        ann_data = ReadMap(dmap_list[k], 'image_info')
        ann_data = ann_data[0, 0][0, 0][0].astype(np.float32).round().astype(int)
        ann_data[:, 0] = ann_data[:, 0].clip(0, img_height - 1)
        ann_data[:, 1] = ann_data[:, 1].clip(0, img_width - 1)

        dmap_data = np.zeros((img_width, img_height), dtype=np.float32)
        dmap_data[(ann_data[:, 1], ann_data[:, 0])] = 1

        # dmap_data = gaussian_filter(dmap_data, sigma=10, mode='constant')

        img_data = img_data.astype('float32')
        dmap_data = dmap_data.astype('float32')

        dmap_data = dmap_data * 100.0
        img_data = img_data / 255.0

        img_clec.append(img_data)
        dmap_clec.append(dmap_data)

    return img_clec, dmap_clec


def load_data_pairs_v1(img_list, dmap_list):
    """load all volume pairs"""
    img_clec = []
    dmap_clec = []

    for k in range(0, len(img_list)):
        img_data = ReadImage(img_list[k])
        dmap_data = ReadMap(dmap_list[k], 'dmap')

        img_data = img_data.astype('float32')
        dmap_data = dmap_data.astype('float32')

        dmap_data = dmap_data * 100.0
        img_data = img_data / 255.0

        img_clec.append(img_data)
        dmap_clec.append(dmap_data)

    return img_clec, dmap_clec

def load_data_pairs_v2(frames):
    """load all volume pairs"""
    img_clec = []
    dmap_clec = []

    for frame in frames:
        img_data = ReadImage(frame.get_image_path())
        dmap_data = frame.get_density()

        img_data = img_data.astype('float32')
        dmap_data = dmap_data.astype('float32')

        dmap_data = dmap_data * 100.0
        img_data = img_data / 255.0

        img_clec.append(img_data)
        dmap_clec.append(dmap_data)

    return img_clec, dmap_clec


def get_batch_patches(rand_img, rand_dmap, patch_dim, batch_size):
    # print rand_img.shape

    if np.random.random() > 0.5:
        rand_img = np.fliplr(rand_img)
        rand_dmap = np.fliplr(rand_dmap)

    w, h, c = rand_img.shape

    patch_width = int(patch_dim[0])
    patch_heigh = int(patch_dim[1])

    batch_img = np.zeros([batch_size, patch_width, patch_heigh, c]).astype('float32')
    batch_dmap = np.zeros([batch_size, patch_width, patch_heigh, 1]).astype('float32')

    rand_img = rand_img.astype('float32')
    rand_dmap = rand_dmap.astype('float32')

    for k in range(batch_size):
        # randomly select a box anchor
        w_rand = randint(0, w - patch_width)
        h_rand = randint(0, h - patch_heigh)

        pos = np.array([w_rand, h_rand])
        # crop
        img_norm = copy.deepcopy(rand_img[pos[0]:pos[0] + patch_width, pos[1]:pos[1] + patch_heigh, :])
        dmap_temp = copy.deepcopy(rand_dmap[pos[0]:pos[0] + patch_width, pos[1]:pos[1] + patch_heigh])

        batch_img[k, :, :, :] = img_norm
        batch_dmap[k, :, :, 0] = dmap_temp

    return batch_img, batch_dmap

def diff_x(input, r):
    assert input.shape.ndims == 4

    left   = input[:,         r:2 * r + 1]
    middle = input[:, 2 * r + 1:         ] - input[:,           :-2 * r - 1]
    right  = input[:,        -1:         ] - input[:, -2 * r - 1:    -r - 1]

    output = tf.concat([left, middle, right], axis=1)

    return output


def diff_y(input, r):
    assert input.shape.ndims == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = tf.concat([left, middle, right], axis=2)

    return output


def box_filter(x, r):
    assert x.shape.ndims == 4
    return diff_y(tf.cumsum(diff_x(tf.cumsum(x, axis=2), r), axis=3), r)
