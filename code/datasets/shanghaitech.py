from glob import glob
import os
import json
import scipy.io as io
import numpy as np

import basic_entities as entities

def load_all_frames(base_path, load_labeling=True):
    frame_paths = glob(os.path.join(base_path, 'images/*.jpg'))
    frames = []

    for img_path in frame_paths:
        frame = entities.BasicFrame(img_path)

        # Load labeling as well
        if load_labeling:
            mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
            gt = mat["image_info"][0,0][0,0][0]

            for point in gt:
                    frame.add_point((int(point[1]), int(point[0])))

        frames.append(frame)

    return frames

if __name__ == '__main__':
    all_frames = load_all_frames('../CSRNet/part_A_final/test_data')
    print(all_frames[0].get_centers())