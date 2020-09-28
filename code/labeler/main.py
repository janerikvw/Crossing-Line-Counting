import os
import inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from datasets import tub, shanghaitech, fudan

import ffmpeg
from glob import glob

for video in glob('../data/Fudan/*/*/'):
    vid = fudan.load_video(video)
    print(vid.get_path())
    ffmpeg.input('{}*.jpg'.format(vid.get_path()), pattern_type='glob', framerate=25).output('{}vid.mp4'.format(vid.get_path())).run()

