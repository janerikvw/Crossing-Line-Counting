import torch
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf

import datasets.tub as tub
import LOI
import utils

from scipy import misc, io
from scipy.ndimage import rotate
import numpy as np
import json
import argparse

from PIL import Image, ImageDraw
import PIL
from tqdm import tqdm

import math

# Configurable params
parser = argparse.ArgumentParser(description='LOI Pipeline')

parser.add_argument('result_dir',metavar='RESULTDIR', type=str,
                    help='Directory where to store everything')

parser.add_argument('--crowd_file', '-c', metavar='CROWDFILE', default='CSRNet/TUB_preShanghaiA-sigma5model_best.pth.tar',type=str,
                    help='Path to pretrained model for crowd counting')

parser.add_argument('--flow_file', '-f', metavar='FLOWFILE', default='DDFlow/Fudan/checkpoints/distillation_census_prekitty2/model-70000',type=str,
                    help='Path to pretrained model for flow estimator')

parser.add_argument('--region_width', '-r', metavar='REGIONWIDTH', default=0.4,type=float,
                    help='Width propotional to the length of the region')

parser.add_argument('--regions', '-s', metavar='REGIONWIDTH', default=5, type=int,
                    help='Amount of regions on each side')


args = parser.parse_args()
args.orig_frame_width = 1280 # Original width of the frame
args.orig_frame_height = 720 # Original height of the frame
# args.regions = 5 # Total amount of regions on each side
args.orientation_shrink = 1.00  # Shrinking per region moving away from the camera (To compensate for orientation)
args.scale = 1. # Scale which we scrink everything to optimize for speed
args.print_time = False # Print every timer, usefull for debugging


# Put all the results in the result directory
result_dir = 'results/{}'.format(args.result_dir)

# Create dir if not exists and create empty results file
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
open("{}/results.txt".format(result_dir), "w+").close()

frame_width, frame_height = utils.scale_frame(args.orig_frame_width, args.orig_frame_height,
                                                               scale=args.scale)

# Loading the complete testset
print("--- LOADING TESTSET ---")
print("- Load videos")
videos = tub.load_all_videos('data/TUBCrowdFlow', load_peds=False)
print('- Loading lines')
video_samples = []
for video in videos:
    base = os.path.basename(video.get_path())
    with open('data/TUBCrowdFlow/crossings/{}.json'.format(base)) as json_file:
        crossing = json.load(json_file)['crossings']
    train_video, train_cross, val_video, val_cross, test_video, test_cross = tub.train_val_test_split(video, crossing)
    video_samples += tub.get_samples_from_video(test_video, test_cross)
print('Total samples:', len(video_samples))
print("--- DONE ---")


# Load counting model
cc_model = LOI.init_cc_model(weights_path=args.crowd_file, img_width=frame_width, img_height=frame_height)
fe_model = None

# Iterate over each video sample
for s_i, (video, line, crosses, naming) in enumerate(video_samples):
    if s_i % 5 == 0:
        tf.reset_default_graph()
        # Load flow estimation model (This to prevent slowing down... Should get fixed pretty soon...
        fe_model = LOI.init_fe_model(restore_model=args.flow_file,
                                     img_width=frame_width, img_height=frame_height)


    print('Sample {}/{}:'.format(s_i+1, len(video_samples)))
    point1 = utils.scale_point(line[0], args.scale)
    point2 = utils.scale_point(line[1], args.scale)
    base = os.path.basename(video.get_path())

    # The width of a single region (args.region_width is propotional to the length of the region)
    line_width = int(math.sqrt(pow(point1[0]-point2[0], 2) + pow(point1[1]-point2[1], 2)) / args.regions * args.region_width)


    timer = utils.sTimer('Initialize LOI')
    loi_model = LOI.init_regionwise_loi(point1, point2,
                                        img_width=frame_width, img_height=frame_height,
                                        line_width=line_width, cregions=args.regions, shrink=args.orientation_shrink)
    timer.show(args.print_time)

    total_left = 0
    total_right = 0

    pbar = tqdm(total=len(video.get_frame_pairs()))

    for i, pair in enumerate(video.get_frame_pairs()):

        print_i = '{:05d}'.format(i+1)

        # Loading images
        timer = utils.sTimer('LI')
        frame1_img = Image.open(pair.get_frames()[0].get_image_path()).convert('RGB').resize((frame_width, frame_height))
        timer.show(args.print_time)
        # frame2_img = Image.open(pair.get_frames()[0].get_image_path()).convert('RGB').resize((frame_width, frame_height))

        # Run counting model on frame A
        timer = utils.sTimer('CC')
        cc_output = LOI.run_cc_model(cc_model, frame1_img.copy())
        timer.show(args.print_time)

        # Run flow estimation model on frame pair
        timer = utils.sTimer('FE')
        fe_output, fe_output_color = LOI.run_fe_model(fe_model, pair)  # Optimize by frame1_img.copy(), frame2_img.copy()
        timer.show(args.print_time)

        # Run the merger for the crowdcounting and flow estimation model
        timer = utils.sTimer('LOI')
        loi_output = LOI.regionwise_loi(loi_model, cc_output, fe_output)
        timer.show(args.print_time)

        # Sum earlier LOI's
        total_left += sum(loi_output[0])
        total_right += sum(loi_output[1])
        totals = [total_left, total_right]

        # Last frame store everything :)
        if i == len(video.get_frame_pairs()) - 1:
            timer = utils.sTimer('Save demo')
            utils.store_processed_images(result_dir, '{}-{}_{}'.format(naming[0], naming[1], s_i), print_i, frame1_img, cc_output, fe_output_color, point1, point2, line_width, args.regions, args.orientation_shrink, loi_output, totals, crosses)
            timer.show(args.print_time)

            result_file = open("{}/results.txt".format(result_dir), "a")
            result_file.write(
                "{}, {}, {}, {}, {}, {}\n".format(naming[0], naming[1], total_left, total_right, float(len(crosses[1])),
                                                  float(len(crosses[0]))))
            result_file.close()

        pbar.set_description('{} ({}), {} ({})'.format(totals[0], len(crosses[1]), totals[1], len(crosses[0])))
        pbar.update(1)



    pbar.close()

    # print('Left to Right: {} ({})'.format(total_left, len(crosses[1])))
    # print('Right to Left: {} ({})'.format(total_right, len(crosses[0])))