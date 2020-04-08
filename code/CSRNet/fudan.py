from glob import glob
import os
import json

import basic_entities as entities

def load_video(video_path):
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()

    video = entities.BasicVideo(video_path)

    for frame_path in frames:
        frame_obj = entities.BasicFrame(frame_path)
        video.add_frame(frame_obj)

        with open(frame_obj.get_image_path().replace('.jpg', '.json')) as json_file:
            data = json.load(json_file)
            data = data[data.keys()[0]]

            for region in data['regions']:
                region = region['shape_attributes']
                frame_obj.add_point(xy=(region['x'], region['y']))

    return video


def load_all_frames(base_path):
    frames = []
    for video_path in glob(os.path.join(base_path, '*')):
        frames = frames + load_video(video_path).get_frames()

    return frames

if __name__ == '__main__':
    train_frames = load_all_frames('../data/Fudan/train_data')
    print(len(train_frames))
