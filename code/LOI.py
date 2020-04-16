import numpy as np
import torch

import datasets
import LOI


from CSRNet.model import CSRNet
import torchvision.transforms.functional as F
import PIL.Image as Image

import tensorflow as tf
from DDFlow.network import pyramid_processing
from DDFlow.flowlib import flow_to_color

def pixelwise_loi(counting_result, flow_result):
    return


def regionwise_loi(counting_result, flow_result):
    return


# Initialize the crowd counting model
def init_cc_model(weights_path = 'CSRNet/fudan_only_model_best.pth.tar'):
    print("--- LOADING CSRNET ---")
    print("- Initialize architecture")
    cc_model = CSRNet()
    print("- Architecture to GPU")
    cc_model = cc_model.cuda()
    print("- Loading weights")
    checkpoint = torch.load(weights_path)
    print("- Weights to GPU")
    cc_model.load_state_dict(checkpoint['state_dict'])
    print("--- DONE ---")
    return cc_model


# Run the crowd counting model on frame A
def run_cc_model(cc_model, frame):
    img = 255.0 * F.to_tensor(Image.open(frame.get_image_path()).convert('RGB'))
    img[0, :, :] = img[0, :, :] - 92.8207477031
    img[1, :, :] = img[1, :, :] - 95.2757037428
    img[2, :, :] = img[2, :, :] - 104.877445883
    img = img.cuda()
    cc_output = cc_model(img.unsqueeze(0))
    return cc_output.detach().cpu().data.numpy().squeeze()


# Initialize the flow estimation model
def init_fe_model(restore_model = 'DDFlow/Fudan/checkpoints/distillation_census_prekitty2/model-70000'):
    print("--- LOADING DDFLOW ---")

    print("- Load architecture")
    frame_img1 = tf.placeholder(tf.float32, [1, None, None, 3], name='img1')
    frame_img2 = tf.placeholder(tf.float32, [1, None, None, 3], name='img2')
    flow_est = pyramid_processing(frame_img1, frame_img2, train=False, trainable=False, regularizer=None, is_scale=True)
    flow_est_color = flow_to_color(flow_est['full_res'], mask=None, max_flow=256)

    opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(var_list=restore_vars)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=opts))

    print("- Initialize architecture")
    sess.run(tf.global_variables_initializer())

    print("- Restore weights")
    saver.restore(sess, restore_model)
    print("--- DONE ---")
    return sess, flow_est, flow_est_color


def run_fe_model(fe_model, pair):
    sess, flow_est, flow_est_color = fe_model

    img1 = tf.image.decode_png(tf.read_file(pair.get_frames()[0].get_image_path()), channels=3)
    img1 = tf.cast(img1, tf.float32)
    img1 /= 255
    img1 = tf.expand_dims(img1, axis=0).eval(session=sess)

    img2 = tf.image.decode_png(tf.read_file(pair.get_frames()[1].get_image_path()), channels=3)
    img2 = tf.cast(img2, tf.float32)
    img2 /= 255
    img2 = tf.expand_dims(img2, axis=0).eval(session=sess)

    np_flow_est, np_flow_est_color = sess.run([flow_est, flow_est_color], feed_dict={'img1:0': img1, 'img2:0': img2})

    return np_flow_est
