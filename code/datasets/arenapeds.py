from glob import glob
import os
import json
import re

import basic_entities as entities


def load_all_videos(videos_path):
    frames = glob(os.path.join(videos_path, '*.jpeg'))
    frames.sort()

    videos = {}
    explode_frame = re.compile("(.*)\/([_0-9]*)_([0-9]*)\.jpeg", re.IGNORECASE)

    for frame_path in frames:
        exploded_frame = explode_frame.match(frame_path)

        frame_obj = entities.BasicFrame(frame_path)

        video_name = exploded_frame.group(2)
        if video_name not in videos:
            videos[video_name] = entities.BasicVideo(videos_path)

        videos[video_name].add_frame(frame_obj)

    return videos


def load_all_frames(base_path):
    videos = load_all_videos(base_path)
    frames = []
    for video_name in videos:
        frames = frames + videos[video_name].get_frames()

    return frames


def load_all_frame_pairs(base_path, load_labeling=True):
    videos = load_all_videos(base_path)
    frames = []
    for video_name in videos:
        frames = frames + videos[video_name].get_frames_pairs()

    return frames

# ############# CURRENTLY MAKES NO SENSE, because no labeling #############
# def load_train_test_frames(base_path, train=0.8, load_labeling=True):
#     videos = load_all_videos(base_path)
#     train_frames = []
#     test_frames = []
#     for video_name in videos:
#         video_frames = load_video(video_path, load_labeling).get_frames()
#         len_frames = len(video_frames)
#         train_frames = train_frames + video_frames[:int(len_frames*train)]
#         test_frames = test_frames + video_frames[int(len_frames * train):]
#
#     return train_frames, test_frames


if __name__ == '__main__':
    train_frames = load_all_frames('../data/Fudan/train_data')
    print(len(train_frames))
