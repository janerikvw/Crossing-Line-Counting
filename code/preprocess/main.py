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

from density_filter import gaussian_filter_density

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
    k = gaussian_filter_density(frame_obj)
    np.save(frame_obj.get_density_path(), k)

if __name__ == '__main__':
    import fudan

    # Loading all the videos based on your dataset in BasicVideo/BasicFrame object format
    videos = []
    for video_path in glob('../data/Fudan/*/*'):
        videos.append(fudan.load_video(video_path))

    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()

    # Start consumers
    num_consumers = multiprocessing.cpu_count() * 2
    print('Creating %d consumers' % num_consumers)
    consumers = [Consumer(tasks) for i in range(num_consumers)]

    # Put every frame in the queue
    for video in videos:
        for frame_obj in video.get_frames():
            tasks.put(Task(frame_obj))

    for w in consumers:
        w.start()

    del videos

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