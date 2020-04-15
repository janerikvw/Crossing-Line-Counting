# Based on the config return the BasicFrames of the specific datasets through a unified function
import fudan
import os

# Return (train_frames, test_frames) with each an array of BasicFrame's to train on.
# These are mainly for training the crowd counting mechanism
def load_train_test_frames(base_path, load_labeling=True):
    return fudan.load_all_frames(os.path.join(base_path, 'data/Fudan/train_data'), load_labeling),\
           fudan.load_all_frames(os.path.join(base_path, 'data/Fudan/test_data'), load_labeling)

# Return (train_pairs, test_pairs) with each an array with BasicPair's to train on
# This method is for training the crowd flow estimation
def load_train_test_frame_pairs(base_path, load_labeling=True):
    return fudan.load_all_frame_pairs(os.path.join(base_path, 'data/Fudan/train_data'), load_labeling),\
           fudan.load_all_frame_pairs(os.path.join(base_path, 'data/Fudan/test_data'), load_labeling)
