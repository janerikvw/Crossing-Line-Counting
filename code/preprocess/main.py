import multiprocessing
import time

import numpy as np
import scipy
from scipy.ndimage.filters import gaussian_filter
import scipy.spatial
import glob
from matplotlib import pyplot as plt
import json
from PIL import ImageDraw, Image
from glob import glob

import matplotlib

from tqdm import tqdm

from density_filter import gaussian_filter_density, gaussian_filter_fixed_density

# Add base path to import dir for importing datasets
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

class Consumer(multiprocessing.Process):
    def __init__(self, task_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break

            next_task()
            self.task_queue.task_done()
        return


class Task(object):
    def __init__(self, frame_info):
        self.info = frame_info

    def __call__(self):
        handle_frame(self.info)

    def __str__(self):
        return self.info.get_image_path()

def handle_frame(frame_obj):
    sizes = [] #  [3,5,8,12,16]

    for size in sizes:
        k = gaussian_filter_fixed_density(frame_obj, sigma=size)
        np.save(frame_obj.get_density_path('fixed-{}'.format(size), check_exists=True), k)

    k = gaussian_filter_density(frame_obj)
    np.save(frame_obj.get_density_path('flex', check_exists=True), k)

if __name__ == '__main__':

    # Loading all the videos based on your dataset in BasicVideo/BasicFrame object format

    from datasets import shanghaitech, fudan, tub
    frames_list = []

    print("Loading ShanghaiTech frames")
    for base_path in glob('../data/ShanghaiTech/part_*/*'):
        frames_list = frames_list + shanghaitech.load_all_frames(base_path)

    print("Loading TUB dataset frames")
    frames_list = frames_list + tub.load_all_frames('../data/TUBCrowdFlow')

    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()

    # Start consumers
    num_consumers = multiprocessing.cpu_count() * 2
    print('Creating %d consumers' % num_consumers)
    consumers = [Consumer(tasks) for i in range(num_consumers)]

    # Put every frame in the queue
    for frame_obj in frames_list:
        tasks.put(Task(frame_obj))

    for w in consumers:
        w.start()

    del frames_list

    # Add a poison pill for each consumer
    for i in range(num_consumers):
        tasks.put(None)

    pbar = tqdm(total=tasks.qsize())

    last_queue = tasks.qsize()
    while tasks.qsize() > 0:
        diff = last_queue - tasks.qsize()
        pbar.update(diff)
        last_queue = tasks.qsize()
        time.sleep(0.2)

    # Wait for all of the tasks to finish
    tasks.join()