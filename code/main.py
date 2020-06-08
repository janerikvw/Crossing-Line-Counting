
import torch
import sys
import os
import tensorflow as tf

# import datasets.factory as factory
# import datasets.fudan as fudan
# import datasets.arenapeds as arenapeds
import datasets.tub as tub
import LOI
import utils

from scipy import misc, io
from scipy.ndimage import rotate
import numpy as np
import json

from PIL import Image, ImageDraw
import PIL
from tqdm import tqdm

# Configurable params
scale = 1.
orig_frame_width = 1280
orig_frame_height = 720

orig_point1 = (550, 20)
orig_point2 = (350, 700)
orig_line_width = 35  # Widest region
orientation_shrink = 1.01  # Shrinking per region moving away from the camera (To compensate for orientation)

result_dir = 'results/new_test'

# Create dir if not exists
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

point1, point2, frame_width, frame_height, line_width = utils.scale_params(orig_point1,
                                                               orig_point2,
                                                               orig_frame_width,
                                                               orig_frame_height,
                                                               orig_line_width,
                                                               scale=scale)

def store_processed_images(sample_name, print_i, frame1_img, cc_output, fe_output_color, point1, point2):
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
    flow_img = utils.white_to_transparency_gradient(flow_img)
    total_image = Image.alpha_composite(total_image, flow_img)

    # Save merged image
    total_image.save(os.path.join(dump_dir, 'combined_{}.png'.format(print_i)))
    img = total_image.convert('RGB')
    # Generate the demo output for clear view on what happens
    utils.image_add_region_lines(img, point1, point2, loi_output, shrink=orientation_shrink, width=line_width)

    # Crop and add information at the bottom of the screen
    utils.add_total_information(img, loi_output, totals, crosses)

    img.save(os.path.join(dump_dir, 'line_{}.png'.format(print_i)))

print("--- LOADING TESTSET ---")
print("- Load videos")
videos = tub.load_all_videos('data/TUBCrowdFlow', load_peds=False)
print('- Loading lines')
video_samples = []
for video in videos:
    base = os.path.basename(video.get_path())
    #crossing = tub.get_line_crossing_frames(video)
    with open('data/TUBCrowdFlow/crossings/{}.json'.format(base)) as json_file:
        crossing = json.load(json_file)['crossings']

    train_video, train_cross, val_video, val_cross, test_video, test_cross = tub.train_val_test_split(video, crossing)
    video_samples += tub.get_samples_from_video(test_video, test_cross)
print('Total samples:', len(video_samples))
print("--- DONE ---")

open("{}/results.txt".format(result_dir),"w+").close()

# Load counting model
# CSRNet/V2_PreB_Fudanmodel_best.pth.tar
cc_model = LOI.init_cc_model(weights_path='CSRNet/TUB_preShanghaiA-sigma5checkpoint.pth.tar', img_width=frame_width, img_height=frame_height)

# Load flow estimation model
fe_model = LOI.init_fe_model(restore_model='DDFlow/Fudan/checkpoints/distillation_census_prekitty2/model-70000', img_width=frame_width, img_height=frame_height)

print_time = False

s_i = 0

for video, line, crosses, naming in video_samples:
    print('Sample {}/{}:'.format(s_i+1, len(video_samples)))
    point1 = line[0]
    point2 = line[1]

    loi_model = LOI.init_regionwise_loi(point1, point2,
                                        img_width=orig_frame_width, img_height=orig_frame_height,
                                        shrink=orientation_shrink, line_width=line_width)

    total_left = 0
    total_right = 0

    pbar = tqdm(total=len(video.get_frame_pairs()))

    base = os.path.basename(video.get_path())
    for i, pair in enumerate(video.get_frame_pairs()):
        print_i = '{:05d}'.format(i+1)

        # Loading images
        frame1_img = Image.open(pair.get_frames()[0].get_image_path()).convert('RGB').resize((frame_width, frame_height))
        # frame2_img = Image.open(pair.get_frames()[0].get_image_path()).convert('RGB').resize((frame_width, frame_height))

        # Run counting model on frame A
        timer = utils.sTimer('CC')
        cc_output = LOI.run_cc_model(cc_model, frame1_img.copy())
        timer.show(print_time)

        # Run flow estimation model on frame pair
        timer = utils.sTimer('FE')
        fe_output, fe_output_color = LOI.run_fe_model(fe_model, pair)  # Optimize by frame1_img.copy(), frame2_img.copy()
        timer.show(print_time)

        # Run the merger for the crowdcounting and flow estimation model
        # In code mention the papers based on
        timer = utils.sTimer('LOI')
        loi_output = LOI.regionwise_loi(loi_model, cc_output, fe_output, fe_output_color)
        timer.show(print_time)

        # Sum earlier LOI's
        to_right = sum(loi_output[1])
        to_left = sum(loi_output[0])
        total_left += to_left
        total_right += to_right
        totals = [total_left, total_right]

        # Last frame store everything :)
        if i == len(video.get_frame_pairs()) - 1:
            timer = utils.sTimer('Save demo')
            store_processed_images('{}-{}_{}'.format(naming[0], naming[1], s_i), print_i, frame1_img, cc_output, fe_output_color, point1, point2)
            timer.show(print_time)

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


    s_i += 1