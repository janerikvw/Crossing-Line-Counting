from glob import glob
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display  # to display images
import os
from datasets import basic_entities as entities
import scipy.io
import pickle
from math import floor

lines = {
    'IM05': [
        # Bottom
        ((600, 400), (610, 719)),
        # Right
        ((800, 300), (1200, 250)),
        # Left
        ((0, 250), (350, 380))
    ],
    'IM02': [
        # Bottom left
        ((320, 400), (500, 450)),
        # Bottom right
        ((550, 400), (900, 470)),
        # Middle
        ((400, 200), (850, 260)),
        # Top
        ((450, 50), (900, 100))

    ],
    'IM01': [
        # Bottom
        ((150, 700), (750, 450)),
        # Top
        ((60, 220), (500, 50))
    ],
    'IM03': [
        # Middle all
        ((600, 550), (640, 250)),
        # Single line, top
        ((800, 380), (810, 300)),
        # Small line, right
        ((1000, 390), (980, 600))
    ],
    'IM04': [
        # Top centre
        ((740, 160), (830, 90)),
        # Left centre
        ((750, 350), (920, 480)),
        # Top
        ((200, 200), (520, 0)),
        # Left
        ((20, 250), (300, 720))
    ]
}

# Load all the videos of the TUB dataset and load the positions of the pedestrians
# @todo Load the tracking of the different pedestrians. (Not required for training right now)
def load_all_videos(path, load_peds=True):
    # Load all the videos from the path and load it into vidoe/frame objects
    videos = []
    for image_dir in glob('{}/images/IM??'.format(path)):
        base = os.path.basename(image_dir)
        frame_files = glob('{}/*'.format(image_dir))
        frame_files.sort()

        video = entities.BasicVideo(image_dir)
        videos.append(video)
        for file_path in frame_files:
            frame_obj = entities.BasicFrame(file_path)
            video.add_frame(frame_obj)

        if base not in lines:
            print("Skipped {}".format(base))
            continue

        frames = video.get_frames()

        # Open pedestrians information
        track_path = '{}/PersonTracks.pb'.format(image_dir.replace('images', 'gt_trajectories'))
        ret = pickle.load(open(track_path, 'rb'))

        if load_peds:
            # Loop per pedestrian over the frames
            for i, pedestrian in enumerate(ret['GT_Trajectories']):
                arr = np.array(pedestrian)
                arr = arr[:, [1, 0]]

                for o, loc in enumerate(arr):
                    frame_id = int(ret['GT_StartPoints'][i][0] + o)
                    frames[frame_id].add_point((loc[0], loc[1]))
                    # TODO: Add tracking, now we haven't connected the track between frames.
    return videos

# Get per video when a pedestrian crosses a line (Get all the information of all the lines of the video)
def get_line_crossing_frames(video):
    # So this function returns a tupple with each an array which gives the frame number that a person crossed a line.
    # This makes it pretty easy to split the long videos into smaller samples
    base = os.path.basename(video.get_path())

    track_path = '{}/PersonTracks.pb'.format(video.get_path().replace('images', 'gt_trajectories'))
    ret = pickle.load(open(track_path, 'rb'))

    video_crosses = {}
    for o, line in enumerate(lines[base]):

        crosses = [[], []]
        for i, pedestrian in enumerate(ret['GT_Trajectories']):

            arr = np.array(pedestrian)
            arr = arr[:, [1, 0]]

            base_point = np.array(line[0])
            vector = np.array(line[1])
            norm_vector = vector - base_point
            norm_arr = arr - base_point

            # Project to check which points fall inside the line
            upper_proj = np.dot(norm_arr, norm_vector)
            lower_proj = np.linalg.norm(norm_vector) ** 2
            proj = upper_proj / lower_proj
            inside = np.array([proj >= 0, proj <= 1]).all(axis=0)

            # Are you on the right side of the line
            check = (norm_vector[0] * norm_arr[:, 1] - norm_vector[1] * norm_arr[:, 0]) > 0;

            # If it didn't cross, then no need to check further
            if np.all(check == check[0]):
                continue

            check2 = np.roll(check, 1)
            together = check[1:] != check2[1:]
            poses = np.where(together == True)[0]

            pos = None
            for pos_i in poses:
                if inside[pos_i] == False:
                    continue

                # Some weird start displacements are filtered out this way
                norm = np.linalg.norm([arr[pos_i, 0] - arr[pos_i + 1, 0], arr[pos_i, 1] - arr[pos_i + 1, 1]])
                if norm > 20:
                    continue

                pos = pos_i

            if pos == None:
                continue

            crossing_frame = int(ret['GT_StartPoints'][i][0] + pos + 2)
            crosses[int(check[pos + 1])].append(crossing_frame)

        print("Loaded line: ", len(crosses[0]), len(crosses[1]))  # 0 == left to right, 1 = right to left
        video_crosses[o] = crosses

    return video_crosses

# Split the video and the crossing information into train and test information
def train_val_test_split(video, crossing_frames, test_size=0.5, train_size=0.1):
    # This function splits the video into train,val and test
    frames = video.get_frames()
    count_frames = len(frames)
    count_train_frames = floor(count_frames * train_size)
    count_test_frames = floor(count_frames * test_size)

    # Get the train video
    train_frames = frames[:count_train_frames]
    train_video = entities.BasicVideo(video.get_path())
    for frame in train_frames:
        train_video.add_frame(frame)

    # Create test video
    test_frames = frames[-count_test_frames:]
    test_video = entities.BasicVideo(video.get_path())
    for frame in test_frames:
        test_video.add_frame(frame)

    # Create validation video. Only add frames if train/test dont overlap
    val_video = entities.BasicVideo(video.get_path())
    if test_size + train_size < 1.0:
        val_frames = frames[count_train_frames:-count_test_frames]
        for frame in val_frames:
            val_video.add_frame(frame)

    train_crossing = {}
    val_crossing = {}
    test_crossing = {}

    # Split the crossing notations for the train/test/val
    for i in crossing_frames:
        train_crossing[i] = [[], []]
        val_crossing[i] = [[], []]
        test_crossing[i] = [[], []]

        for o, crossing_side in enumerate(crossing_frames[i]):
            for side_jump in crossing_side:
                if side_jump < count_train_frames:
                    train_crossing[i][o].append(side_jump)
                elif side_jump >= (count_frames - count_test_frames):
                    test_crossing[i][o].append(side_jump - (count_frames - count_test_frames))
                else:
                    val_crossing[i][o].append(side_jump - count_train_frames)

    # Clean val if test/train only
    if len(val_video.get_frames()) == 0:
        val_crossing = None
        val_video = None

    return train_video, train_crossing, val_video, val_crossing, test_video, test_crossing

# Split a video (So probably only a test/validation video) into multiple shorter samples.
# Sample_overlap is the amount of frames the begin of sample 1 and begin of sample 2 are apart.
def get_samples_from_video(video, crossing_frames, sample_length=50, sample_overlap=50):
    # Split the long video into shorter samples and per line
    base = os.path.basename(video.get_path())
    count_frames = len(video.get_frames())

    if base not in lines:
        print('{} has not lines'.format(base))
        return

    video_samples = []
    for line_i in crossing_frames:
        line_crosses = crossing_frames[line_i]

        # Extract from the validation part, take smaller parts as evaluation sample
        for o in range(0, count_frames - sample_overlap + 1, sample_overlap):
            start_sample = o
            end_sample = min(o + sample_length, count_frames)
            # Check all the original frame jumps and take only the ones with this sample
            check_line_crosses = []
            for line_cross in line_crosses:
                valid_line_cross = []
                for frame_jump in line_cross:
                    if frame_jump >= start_sample and frame_jump < end_sample:
                        valid_line_cross.append(frame_jump)
                check_line_crosses.append(valid_line_cross)

            video_samples.append([start_sample, end_sample, lines[base][line_i], check_line_crosses])

    return video_samples