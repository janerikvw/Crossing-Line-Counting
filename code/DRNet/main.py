import os
import tensorflow as tf
import numpy as np
import random

from tensorflow.python import debug as tf_debug

from ini_file_io import load_train_ini
from model import drnet_2D

# set cuda visable device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def main(_):
    tf.reset_default_graph()
    # load training parameter #
    ini_file = './ini/tr_param.ini'
    param_sets = load_train_ini(ini_file)
    param_set = param_sets[0]

    print('====== Phase >>> %s <<< ======' % param_set['phase'])

    if not os.path.exists(param_set['chkpoint_dir']):
        os.makedirs(param_set['chkpoint_dir'])
    if not os.path.exists(param_set['labeling_dir']):
        os.makedirs(param_set['labeling_dir'])

    set_random_seed(31480915)
    # GPU setting, per_process_gpu_memory_fraction means 95% GPU MEM ,allow_growth means unfixed memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        model = drnet_2D(sess, param_set)

        if param_set['phase'] == 'train':
            model.train()
        elif param_set['phase'] == 'test':
            model.test()
        elif param_set['phase'] == 'inference':
            img_path = '../data/ShanghaiTech/part_A/test/img/IMG_7.jpg'
            dmap_path = '../data/ShanghaiTech/part_A/test/ann/GT_IMG_7.mat'
            model.inference(img_path, dmap_path)
        elif param_set['phase'] == 'visualize':
            model.visualize()

if __name__ == '__main__':
    tf.app.run()
