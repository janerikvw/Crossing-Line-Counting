import torch
import sys
import os

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

from DDFlow_pytorch.utils import flo_to_color

import predictors
import dataloaders

def load_videos(args):
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
        _, _, _, _, test_video, test_cross = tub.train_val_test_split(video, crossing)

        video_samples += tub.get_samples_from_video(test_video, test_cross)
    print('Total samples:', len(video_samples))
    print("--- DONE ---")
    return video_samples


def load_dataloader(video, args):
    video.generate_frame_pairs(distance=args.pair_distance, skip_inbetween=True)

    return dataloaders.PairDataloader(video.get_frame_pairs(),
                                      img_width=args.frame_width, img_height=args.frame_height)



def main(args):
    # Create dir if not exists and create empty results file
    if not os.path.exists(args.full_dir):
        os.mkdir(args.full_dir)
    open("{}/results.txt".format(args.full_dir), "w+").close()

    video_samples = load_videos(args)

    if ARGS.merged_model:
        model = predictors.FullPredictor()
    else:
        cc = predictors.CSRPredictor()
        fe = predictors.PWCPredictor()
        model = predictors.CombinedPredictor(cc, fe)

    for s_i, (video, line, crosses, naming) in enumerate(video_samples):
        # Start with 0 counting
        total_left = 0
        total_right = 0

        print('Sample {}/{}:'.format(s_i + 1, len(video_samples)))
        point1 = line[0]
        point2 = line[1]
        ped_size = line[2]
        base = os.path.basename(video.get_path())

        timer = utils.sTimer('Initialize LOI')
        loi_model = predictors.RegionLOI(point1, point2, img_width=args.frame_width, img_height=args.frame_height,
                                         ped_size=ped_size, width_peds=args.width_times,
                                         height_peds=args.height_times, crop_processing=ARGS.cropping)
        timer.show(args.print_time)

        dataloader = load_dataloader(video, args)

        pbar = tqdm(total=len(dataloader))

        for i, (frame1, frame2) in enumerate(dataloader):
            print(i, frame1.shape, frame2.shape)
            exit()

if __name__ == '__main__':
    # Configurable params
    parser = argparse.ArgumentParser(description='LOI Pipeline')

    parser.add_argument('result_dir', metavar='RESULTDIR', type=str,
                        help='Directory where to store everything')

    parser.add_argument('--crowd_file', '-c', metavar='CROWDFILE',
                        default='CSRNet/TUB_preShanghaiA-sigma5model_best.pth.tar', type=str,
                        help='Path to pretrained model for crowd counting')

    parser.add_argument('--flow_file', '-f', metavar='FLOWFILE',
                        default='DDFlow_pytorch/weights/20200622_111127_train2_v1/model_150000.pt', type=str,
                        help='Path to pretrained model for flow estimator')

    parser.add_argument('--full_file', '-g', metavar='FLOWFILE',
                        default='full_on_pwc/weights/20200728_110816_full_test/best_model.pt', type=str,
                        help='Path to pretrained model for flow estimator')

    parser.add_argument('--width_times', '-w', metavar='WIDTHPEDS', default=3.0, type=float,
                        help='The width of each region times the pedestrian size')

    parser.add_argument('--height_times', '-t', metavar='HEIGHTPEDS', default=2.0, type=float,
                        help='The height of each region times the pedestrian size')

    ARGS = parser.parse_args()
    ARGS.frame_width = 1280  # Original width of the frame
    ARGS.frame_height = 720  # Original height of the frame

    ARGS.print_time = False  # Print every timer, usefull for debugging

    ARGS.pair_distance = 1  # Distance between frames (normally 1 for next, 25fps for TUB)

    ARGS.region_select = 'V2'  # V1 or V2 for comparison
    ARGS.cropping = 100  # Cropping for quicker processing (Give number as padding for outers to optimize performance)

    ARGS.merged_model = False
    ARGS.full_dir = 'results_new/{}'.format(ARGS.result_dir)

    main(ARGS)