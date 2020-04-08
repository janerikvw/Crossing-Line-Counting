##############################################
############# SHOULD UPDATE ##################
##############################################

# TODO: Update
from glob import glob
import scipy.io

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.animation as animation
from IPython.core.display import display, HTML
from tqdm.notebook import tqdm
from IPython.display import clear_output
from PIL import ImageDraw, Image
import cv2
import re
import itertools


import ntpath


def get_all_frame_files(path, vid_dir):
    tracked_files = glob('{}vidf1_*_people_full.mat'.format(path))

    ret = []

    for file in tracked_files:
        grabbed_name = re.search('(vidf1_33\_.*)\_people', ntpath.basename(file)).group(1)
        video_dir = '{}{}.y/'.format(vid_dir, grabbed_name)
        if len(glob('{}*'.format(video_dir))) == 0:
            # print("Dammit where is the vidf1 directory?")
            continue

        ret.append([
            file,
            video_dir
        ])
    return ret

def get_video_info(tracked_file, video_dir, track_id = False):
    frames_names = glob('{}*.png'.format(video_dir))
    frames_names.sort()

    if track_id:
        frame_info = [{'filename': name, 'persons': {}} for name in frames_names]
    else:
        frame_info = [{'filename': name, 'persons': []} for name in frames_names]

    mat = scipy.io.loadmat(tracked_file)
    for idx, person in enumerate(mat['people'][0]):

        for coordinates in person[0][0][0]:
            if track_id:
                frame_info[int(coordinates[2]) - 1]['persons'][idx] = {'x': float(coordinates[0]),
                                                                   'y': float(coordinates[1])}
            else:
                frame_info[int(coordinates[2]) - 1]['persons'].append([float(coordinates[0]), float(coordinates[1])])

    return frame_info


