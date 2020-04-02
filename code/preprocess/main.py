import multiprocessing
import time

import numpy as np
import scipy
from scipy.ndimage.filters import gaussian_filter
import scipy.spatial
import glob
from matplotlib import pyplot as plt
import json
import ucsdpeds
from PIL import ImageDraw, Image

import matplotlib

from tqdm import tqdm

#partly borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(img,points):
    '''
    This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.

    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.

    return:
    density: the density-map we want. Same shape as input image but only has one channel.

    example:
    points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
    img_shape: (768,1024) 768 is row and 1024 is column.
    '''
    img_shape=[img.shape[0],img.shape[1]]
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)

    # query kdtree
    distances, locations = tree.query(points, k=4)

    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2.
        density += gaussian_filter(pt2d, sigma, mode='constant', truncate=3.0)

    return density

def handle_frame(frame_info):
    img = plt.imread(frame_info['filename'])
    k = gaussian_filter_density(img, np.array(frame_info['persons']))
    np.save(frame_info['filename'].replace('.png', '.npy'), k)



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
                #print('%s: Exiting' % proc_name)
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
        return self.info['filename']


if __name__ == '__main__':

    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()

    # Start consumers
    num_consumers = multiprocessing.cpu_count() * 2
    print('Creating %d consumers' % num_consumers)
    consumers = [Consumer(tasks) for i in range(num_consumers)]
    for w in consumers:
        w.start()

    all_frame_files = ucsdpeds.get_all_frame_files('../data/ucsdpeds/gt/vidf/', '../data/ucsdpeds/vidf/')
    # Put every frame in the queue
    for movie_path in all_frame_files:
        all_frame_info = ucsdpeds.get_video_info(movie_path[0], movie_path[1])

        for frame_info in all_frame_info:
            tasks.put(Task(frame_info))

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