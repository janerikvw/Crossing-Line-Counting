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

def main(args):
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
    if ARGS.merged_model:
        full_model = LOI.init_full_model(weights_path=args.full_file)
    else:
        cc_model = LOI.init_cc_model(weights_path=args.crowd_file, img_width=frame_width, img_height=frame_height)
        fe_model = LOI.init_fe_model(weights_path=args.flow_file, img_width=frame_width, img_height=frame_height)

    # Iterate over each video sample
    import random
    #random.shuffle(video_samples)

    for s_i, (video, line, crosses, naming) in enumerate(video_samples):
        print('Sample {}/{}:'.format(s_i+1, len(video_samples)))
        point1 = utils.scale_point(line[0], args.scale)
        point2 = utils.scale_point(line[1], args.scale)
        ped_size = utils.scale_point(line[2], args.scale)
        base = os.path.basename(video.get_path())

        # The width of a single region (args.region_width is propotional to the length of the region)


        timer = utils.sTimer('Initialize LOI')
        loi_model = LOI.init_regionwise_loi(point1, point2,
                                            img_width=frame_width, img_height=frame_height, ped_size=ped_size,
                                            width_peds=args.width_times, height_peds=args.height_times,
                                            select_type=ARGS.region_select, crop_processing=ARGS.cropping)
        timer.show(args.print_time)

        total_left = 0
        total_right = 0

        video.generate_frame_pairs(distance=args.pair_distance, skip_inbetween=True)
        pbar = tqdm(total=len(video.get_frame_pairs()))

        for i, pair in enumerate(video.get_frame_pairs()):
            print_i = '{:05d}'.format(i+1)

            # Loading images
            timer = utils.sTimer('LI')
            frame1_img = Image.open(pair.get_frames()[0].get_image_path()).resize((frame_width, frame_height))
            frame2_img = Image.open(pair.get_frames()[1].get_image_path()).resize((frame_width, frame_height))
            frame1_img = LOI.preprocess_image_loi(loi_model, frame1_img)
            frame2_img = LOI.preprocess_image_loi(loi_model, frame2_img)

            timer.show(args.print_time)

            if ARGS.merged_model:
                timer = utils.sTimer('Full pass')
                fe_output, cc_output = LOI.run_full_model(full_model, frame1_img, frame2_img)
                timer.show(args.print_time)
            else:
                # Run counting model on frame A
                timer = utils.sTimer('CC')
                cc_output = LOI.run_cc_model(cc_model, frame1_img.copy())
                timer.show(args.print_time)

                # Run flow estimation model on frame pair
                timer = utils.sTimer('FE')
                fe_output = LOI.run_fe_model(fe_model, frame1_img, frame2_img)
                timer.show(args.print_time)

            # Run the merger for the crowdcounting and flow estimation model
            timer = utils.sTimer('LOI')
            if args.loi_method =='region':
                loi_output = LOI.regionwise_loi(loi_model, cc_output, fe_output)
            else:
                loi_output = LOI.pixelwise_loi(loi_model, cc_output, fe_output)
            timer.show(args.print_time)

            # Sum earlier LOI's
            total_left += sum(loi_output[0])
            total_right += sum(loi_output[1])
            totals = [total_left, total_right]


            # Last frame store everything :)
            if i == len(video.get_frame_pairs()) - 1:

                timer = utils.sTimer('Save demo')
                fe_output_color = flo_to_color(fe_output)
                utils.store_processed_images(result_dir, '{}-{}_{}'.format(naming[0], naming[1], s_i), print_i,
                                             frame1_img,
                                             cc_output, fe_output_color, point1, point2, loi_model, loi_output, totals,
                                             crosses)
                timer.show(args.print_time)

                result_file = open("{}/results.txt".format(result_dir), "a")
                result_file.write(
                    "{}, {}, {}, {}, {}, {}\n".format(naming[0], naming[1], total_left, total_right, float(len(crosses[1])),
                                                      float(len(crosses[0]))))
                result_file.close()
            pbar.set_description('{} ({}), {} ({})'.format(totals[0], len(crosses[1]), totals[1], len(crosses[0])))
            pbar.update(1)

        pbar.close()


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
    ARGS.orig_frame_width = 1280  # Original width of the frame
    ARGS.orig_frame_height = 720  # Original height of the frame

    ARGS.scale = 1.  # Scale which we scrink everything to optimize for speed
    ARGS.print_time = False  # Print every timer, usefull for debugging

    ARGS.pair_distance = 5  # Distance between frames (normally 1 for next, 25fps for TUB)
    ARGS.loi_method = 'region'  # pixel or region wise

    ARGS.region_select = 'V2'  # V1 or V2 for comparison
    ARGS.cropping = 100  # Cropping for quicker processing (Give number as padding for outers to optimize performance)

    ARGS.merged_model = False

    main(ARGS)