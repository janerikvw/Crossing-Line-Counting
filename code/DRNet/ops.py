import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn
import numpy as np
from utils import box_filter


#######################
# 3d functions
#######################

def batch_normalization(input, name):
    with tf.variable_scope(name):
        return tf.contrib.layers.instance_norm(input)


# convolution
def conv2d(input, output_chn, kernel_size, stride, dilation, use_bias=False, name='conv'):
    return tf.layers.conv2d(inputs=input, filters=output_chn, kernel_size=kernel_size, strides=stride,
                            dilation_rate=dilation, padding='same', data_format='channels_last',
                            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            kernel_regularizer=slim.l2_regularizer(0.0005), use_bias=use_bias, name=name)


def conv_bn_relu(input, output_chn, kernel_size, stride, dilation, use_bias, name):
    with tf.variable_scope(name):
        conv = conv2d(input, output_chn, kernel_size, stride, dilation, use_bias, name='conv')
        bn = batch_normalization(conv, name="batch_norm")
        relu = tf.nn.relu(bn, name='relu')

    return relu


def conv_relu(input, output_chn, kernel_size, stride, dilation, use_bias, name):
    with tf.variable_scope(name):
        conv = conv2d(input, output_chn, kernel_size, stride, dilation, use_bias, name='conv')
        relu = tf.nn.relu(conv, name='relu')

    return relu


# deconvolution
def Deconv2d(input, output_chn, kernel_size, stride, name):
    # batch, in_height, in_width, in_channels = [int(d) for d in input.get_shape()]
    static_input_shape = input.get_shape().as_list()
    dyn_input_shape = tf.shape(input)
    filter = tf.get_variable(name + "/filter", shape=[kernel_size, kernel_size, output_chn, static_input_shape[3]],
                             dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0, 0.01), regularizer=slim.l2_regularizer(0.0005))

    conv = tf.nn.conv2d_transpose(value=input, filter=filter,
                                  output_shape=[dyn_input_shape[0], dyn_input_shape[1] * stride,
                                                dyn_input_shape[2] * stride, output_chn],
                                  strides=[1, stride, stride, 1], padding="SAME", name=name)
    return conv


def deconv_bn_relu(input, output_chn, kernel_size, stride, name):
    with tf.variable_scope(name):
        conv = Deconv2d(input, output_chn, kernel_size, stride, name='deconv')
        bn = batch_normalization(conv, name="batch_norm")
        relu = tf.nn.relu(bn, name='relu')
    return relu


def conv_bn_relu_x2(input, output_chn, kernel_size, stride, dilation, use_bias, name):
    with tf.variable_scope(name):
        z = conv_bn_relu(input, output_chn, kernel_size, stride, dilation, use_bias, "dense1")
        z_out = conv_bn_relu(z, output_chn, kernel_size, stride, dilation, use_bias, "dense2")
        return z_out


def conv_bn_relu_x3(input, output_chn, kernel_size, stride, dilation, use_bias, name):
    with tf.variable_scope(name):
        z = conv_bn_relu(input, output_chn, kernel_size, stride, dilation, use_bias, "dense1")
        z_temp = conv_bn_relu(z, output_chn, kernel_size, stride, dilation, use_bias, "dense2")
        z_out = conv_bn_relu(z_temp, output_chn, kernel_size, stride, dilation, use_bias, "dense3")
        return z_out


def bottleneck_block(input, input_chn, output_chn, kernel_size, stride, dilation, use_bias, name):
    with tf.variable_scope(name):
        # Convolutional Layer 1
        layer_conv1 = conv_bn_relu(input=input, output_chn=output_chn, kernel_size=kernel_size, stride=stride,
                                   dilation=dilation, use_bias=use_bias, name=name + '_conv1')
        # Convolutional Layer 2
        layer_conv2 = conv2d(input=layer_conv1, output_chn=output_chn, kernel_size=kernel_size, stride=1,
                             dilation=dilation, use_bias=use_bias, name=name + '_conv2')
        bn = batch_normalization(layer_conv2, name="batch_norm")
        if input_chn == output_chn and stride == 1:
            res = bn + input
        else:
            layer_conv3 = conv2d(input=input, output_chn=output_chn, kernel_size=1, stride=stride, dilation=(1, 1),
                                 use_bias=use_bias, name=name + '_conv3')
            bn1 = batch_normalization(layer_conv3, name="batch_norm1")
            res = bn + bn1

        return tf.nn.relu(res, name='relu')


def recursive_box_filter(input, K=2, r=8):
    img_bf = box_filter(input, r)
    for i in xrange(0, K):
        img_bf = box_filter(img_bf, r)

    return img_bf


def extract_keypoints(heatmap, score_threshold=0.05):
    keypoints = heatmap / 100.0
    # keypoints = tf.sigmoid(heatmap)
    keypoints_peak = tf.layers.max_pooling2d(keypoints, 3, 1, 'SAME')
    keypoints_mask = tf.cast(tf.equal(keypoints, keypoints_peak), tf.float32)
    keypoints = keypoints * keypoints_mask
    keypoints = tf.cast(tf.greater_equal(keypoints, score_threshold), tf.float32)

    return keypoints