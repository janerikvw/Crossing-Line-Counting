from __future__ import division
import os
import time
from glob import glob
from ops import *
from utils import *
import numpy as np
import copy
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.measure import compare_psnr, compare_ssim
from scipy import misc
import numpy as np
from guided_filter_new import guided_filter

import datetime


class drnet_2D(object):

    def __init__(self, sess, param_set):
        self.sess = sess
        #self.phase = param_set['phase']
        self.batch_size = param_set['batch_size']
        self.inputI_width_size = param_set['inputI_width_size']
        self.inputI_height_size = param_set['inputI_height_size']
        self.inputI_chn = param_set['inputI_chn']
        self.output_chn = param_set['output_chn']
        self.trainImagePath = param_set['trainImagePath']
        self.trainDmapPath = param_set['trainDmapPath']
        self.testImagePath = param_set['testImagePath']
        self.testDmapPath = param_set['testDmapPath']
        self.chkpoint_dir = param_set['chkpoint_dir']
        self.lr = param_set['learning_rate']
        # self.beta1 = param_set['beta1']
        self.epoch = param_set['epoch']
        self.model_name = param_set['model_name']
        self.load_model_path = param_set['load_model_path']
        # self.save_intval = param_set['save_intval']
        # self.labeling_dir = param_set['labeling_dir']

        self.inputI_size = [self.inputI_width_size, self.inputI_height_size]
        # build model graph
        self.build_model()

    def l1_loss(self, prediction, ground_truth, weight_map=None):
        """
        :param prediction: the current prediction of the ground truth.
        :param ground_truth: the measurement you are approximating with regression.
        :return: mean of the l1 loss across all voxels.
        """
        absolute_residuals = tf.abs(tf.subtract(prediction, ground_truth))
        return tf.reduce_sum(tf.sqrt(absolute_residuals**2+1e-6))

    def l2_loss(self, prediction, ground_truth):
        """
        :param prediction: the current prediction of the ground truth.
        :param ground_truth: the measurement you are approximating with regression.
        :return: sum(differences squared) / 2 - Note, no square root
        """

        residuals = tf.abs(tf.subtract(prediction, ground_truth))
        return tf.nn.l2_loss(residuals)


    def focal_loss_func(self, logits, labels, alpha=0.25, gamma=2.0):
        """
        Loss = weighted * -target*log(softmax(logits))
        :param logits: probability score
        :param labels: ground_truth
        :return: softmax-weighted loss
        """

        labels = tf.dtypes.cast(labels[:, :, :, 0], tf.int32)
        gt = tf.one_hot(labels, 2)
        softmaxpred = tf.nn.softmax(logits)
        loss = 0
        for i in range(2):
            gti = gt[:, :, :, i]
            predi = softmaxpred[:, :, :, i]
            weighted = 1 - (tf.reduce_sum(gti) / tf.reduce_sum(gt))
            loss = loss + tf.reduce_mean(
                weighted * gti * tf.pow(1 - predi, gamma) * tf.log(tf.clip_by_value(predi, 0.005, 1)))
        return -loss / 2

    # build model graph
    def build_model(self):
        # input
        self.input_Img = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.inputI_chn], name='input_Img')
        self.input_Dmap = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.output_chn], name='input_Dmap')

        print('Model: drnet_D_2D_model')
        self.pred_prob = self.drnet_D_2D_model(self.input_Img)

        self.density_loss = self.l1_loss(self.pred_prob, self.input_Dmap)
        # self.density_loss = self.l2_loss(recursive_box_filter(self.pred_prob), recursive_box_filter(self.input_Dmap))

        self.total_loss = self.density_loss
        # trainable variables
        self.u_vars = tf.trainable_variables()

        # create model saver
        self.saver = tf.train.Saver(max_to_keep=1000)

    def drnet_D_2D_model(self, inputI):
        concat_dim = 3
        chn = 64

        # ***************encoder level0***************
        conv1 = conv_bn_relu(input=inputI, output_chn=chn, kernel_size=7, stride=1, dilation=(1, 1), use_bias=False,
                             name='conv1')
        print("output of encoder level0:")
        print(conv1.get_shape())

        # ***************encoder level1***************
        res_block1 = bottleneck_block(input=conv1, input_chn=chn, output_chn=chn, kernel_size=3, stride=2,
                                      dilation=(1, 1), use_bias=False, name='res_block1')
        res_block1 = bottleneck_block(input=res_block1, input_chn=chn, output_chn=chn, kernel_size=3, stride=1,
                                      dilation=(1, 1), use_bias=False, name='res_block1_1')
        print("output of encoder level2:")
        print(res_block1.get_shape())

        # *************encoder level2***************
        res_block2 = bottleneck_block(input=res_block1, input_chn=chn, output_chn=chn, kernel_size=3, stride=2,
                                      dilation=(1, 1), use_bias=False, name='res_block2')
        res_block2 = bottleneck_block(input=res_block2, input_chn=chn, output_chn=chn, kernel_size=3, stride=1,
                                      dilation=(1, 1), use_bias=False, name='res_block2_1')
        print("output of encoder level3:")
        print(res_block2.get_shape())

        # *************encoder level3***************
        res_block3 = bottleneck_block(input=res_block2, input_chn=chn, output_chn=chn, kernel_size=3, stride=1,
                                      dilation=(2, 2), use_bias=False, name='res_block3')
        res_block3 = bottleneck_block(input=res_block3, input_chn=chn, output_chn=chn, kernel_size=3, stride=1,
                                      dilation=(2, 2), use_bias=False, name='res_block3_1')
        print("output of encoder level4:")
        print(res_block3.get_shape())

        # *************encoder level4***************
        res_block4 = bottleneck_block(input=res_block3, input_chn=chn, output_chn=chn, kernel_size=3, stride=1,
                                      dilation=(4, 4), use_bias=False, name='res_block4')
        res_block4 = bottleneck_block(input=res_block4, input_chn=chn, output_chn=chn, kernel_size=3, stride=1,
                                      dilation=(4, 4), use_bias=False, name='res_block4_1')
        print("output of encoder level5:")
        print(res_block4.get_shape())

        # *************encoder level5***************
        res_block5 = bottleneck_block(input=res_block4, input_chn=chn, output_chn=chn, kernel_size=3, stride=1,
                                      dilation=(2, 2), use_bias=False, name='res_block5')
        res_block5 = bottleneck_block(input=res_block5, input_chn=chn, output_chn=chn, kernel_size=3, stride=1,
                                      dilation=(2, 2), use_bias=False, name='res_block5_1')
        print("output of encoder level6:")
        print(res_block5.get_shape())

        # *************encoder level6***************
        res_block6 = bottleneck_block(input=res_block5, input_chn=chn, output_chn=chn, kernel_size=3, stride=1,
                                      dilation=(1, 1), use_bias=False, name='res_block6')
        res_block6 = bottleneck_block(input=res_block6, input_chn=chn, output_chn=chn, kernel_size=3, stride=1,
                                      dilation=(1, 1), use_bias=False, name='res_block6_1')
        print("output of decoder level7:")
        print(res_block6.get_shape())

        # *************encoder level7***************
        # concat1 = tf.concat([res_block7, res_block5], axis=concat_dim, name='concat1')
        res_block7 = bottleneck_block(input=res_block6, input_chn=chn, output_chn=chn, kernel_size=3, stride=1,
                                      dilation=(1, 1), use_bias=False, name='res_block7')
        res_block7 = bottleneck_block(input=res_block7, input_chn=chn, output_chn=chn, kernel_size=3, stride=1,
                                      dilation=(1, 1), use_bias=False, name='res_block7_1')
        print("output of decoder level8:")
        print(res_block7.get_shape())

        # *************encoder level8***************
        # concat2 = tf.concat([res_block8, res_block4], axis=concat_dim, name='concat2')
        res_block8 = conv_bn_relu_x2(input=res_block7, output_chn=chn, kernel_size=3, stride=1, dilation=(1, 1),
                                     use_bias=False, name='res_block8')
        res_block8 = conv_bn_relu_x2(input=res_block8, output_chn=chn, kernel_size=3, stride=1, dilation=(1, 1),
                                     use_bias=False, name='res_block8_1')
        print("output of decoder level9:")
        print(res_block8.get_shape())

        # *************decoder level10***************
        deconv1_upsample = deconv_bn_relu(input=res_block8, output_chn=chn, kernel_size=4, stride=2,
                                          name='deconv1_upsample')
        deconv1_conv1 = conv_bn_relu_x2(input=deconv1_upsample, output_chn=chn, kernel_size=3, stride=1,
                                        dilation=(1, 1), use_bias=False, name='deconv1_conv1')
        print("output of decoder level10:")
        print(deconv1_conv1.get_shape())

        # *************decoder level11***************
        deconv2_upsample = deconv_bn_relu(input=deconv1_conv1, output_chn=chn, kernel_size=4, stride=2,
                                          name='deconv2_upsample')
        deconv2_conv1 = conv_bn_relu_x2(input=deconv2_upsample, output_chn=chn, kernel_size=3, stride=1,
                                        dilation=(1, 1), use_bias=False, name='deconv2_conv1')
        print("output of decoder level11:")
        print(deconv2_conv1.get_shape())

        pred_prob_temp = conv2d(input=deconv2_conv1, output_chn=1, kernel_size=1, stride=1, dilation=(1, 1), use_bias=True,
                           name='pred_prob')

        self.test1 = pred_prob_temp

        guide_prob_temp = conv2d(input=conv1, output_chn=1, kernel_size=3, stride=1, dilation=(1,1), use_bias=True, name='guide_prob_temp')
        pred_prob = guided_filter(guide_prob_temp, pred_prob_temp, 4, eps=1e-2, nhwc=True)
        self.test2 = pred_prob

        return pred_prob

    # train function
    def train(self):

        u_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.total_loss, var_list=self.u_vars)

        # initialization
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        log_file = open("./results/" + self.model_name + "_log.txt", "w")

        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS\n")
            log_file.write(" [*] Load SUCCESS\n")
        else:
            print(" [!] Load failed...\n")
            log_file.write(" [!] Load failed...\n")

        print("Load data")

        # # load all volume files
        # img_list = glob('{}/*.jpg'.format(self.trainImagePath))
        # img_list.sort()
        # dmap_list = glob('{}/*.mat'.format(self.trainDmapPath))
        # dmap_list.sort()
        # img_clec, dmap_clec = load_data_pairs(img_list, dmap_list)

        import os, sys, inspect
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(currentdir)
        sys.path.insert(0, parentdir)
        from datasets import shanghaitech, fudan
        # frames = shanghaitech.load_all_frames('../data/ShanghaiTech/part_B_final/train_data', load_labeling=False)
        train_frames, test_frames = fudan.load_train_test_frames('../data/Fudan/train_data')
        import random
        random.shuffle(train_frames)
        random.shuffle(test_frames)
        train_frames = train_frames[0:800]
        test_frames = test_frames[0:150]
        img_clec, dmap_clec = load_data_pairs_v2(train_frames)
        test_img_clec, test_dmap_clec = load_data_pairs_v2(test_frames)

        # # get file list of testing dataset
        # test_img_list = glob('{}/*.jpg'.format(self.testImagePath))
        # test_img_list.sort()
        # test_dmap_list = glob('{}/*.mat'.format(self.testDmapPath))
        # test_dmap_list.sort()
        # test_img_clec, test_dmap_clec = load_data_pairs(test_img_list, test_dmap_list)

        # frames = shanghaitech.load_all_frames('../data/ShanghaiTech/part_B_final/test_data', load_labeling=False)

        self.test_training(test_img_clec, test_dmap_clec, 0, log_file)

        rand_idx = np.arange(len(img_clec))
        start_time = time.time()
        # count=0
        for epoch in np.arange(self.epoch):
            np.random.shuffle(rand_idx)
            epoch_total_loss = 0.0
            for i_dx in rand_idx:  # xrange(len(img_clec)):
                # train batch
                batch_img, batch_dmap = get_batch_patches(img_clec[i_dx], dmap_clec[i_dx], self.inputI_size,
                                                          self.batch_size)

                _, cur_train_loss = self.sess.run([u_optimizer, self.total_loss],
                                                  feed_dict={self.input_Img: batch_img, self.input_Dmap: batch_dmap})

                epoch_total_loss += cur_train_loss

            # if np.mod(epoch+1, 2) == 0:
            print("Epoch: [%d] time: %4.4f, train_loss: %.8f\n" % (
            epoch + 1, time.time() - start_time, epoch_total_loss / len(img_clec)))
            log_file.write("Epoch: [%d] time: %4.4f, train_loss: %.8f\n" % (
            epoch + 1, time.time() - start_time, epoch_total_loss / len(img_clec)))
            log_file.flush()
            start_time = time.time()

            if epoch + 1 > 0:  # np.mod(epoch+1, self.save_intval) == 0:
                self.test_training(test_img_clec, test_dmap_clec, epoch + 1, log_file)
                self.save_chkpoint(self.chkpoint_dir, self.model_name, epoch + 1)

        log_file.close()

    def test_training(self, test_img_clec, test_dmap_clec, step, log_file):
        all_mae = np.zeros([len(test_img_clec)])
        all_rmse = np.zeros([len(test_img_clec)])

        for k in range(0, len(test_img_clec)):
            # print k
            img_data = test_img_clec[k]

            w, h, c = img_data.shape
            w = int(w / 4) * 4
            h = int(h / 4) * 4

            img_data = resize(img_data, (w, h, c), preserve_range=True)
            img_data = img_data.reshape(1, w, h, c)

            dmap_data = test_dmap_clec[k] / 100.0

            predicted_label = self.sess.run(self.pred_prob, feed_dict={self.input_Img: img_data})
            predicted_label /= 100.0

            all_mae[k] = abs(np.sum(predicted_label) - np.sum(dmap_data))
            all_rmse[k] = pow((np.sum(predicted_label) - np.sum(dmap_data)), 2)

        mean_mae = np.mean(all_mae, axis=0)
        mean_rmse = pow(np.mean(all_rmse, axis=0), 0.5)
        print("Epoch: [%d], mae: %0.2f, rmse:%0.2f\n" % (step, mean_mae, mean_rmse))
        log_file.write("Epoch: [%d], mae: %0.2f, rmse:%0.2f\n" % (step, mean_mae, mean_rmse))
        log_file.flush()

    def test(self):

        # initialization
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS\n")
        else:
            print(" [!] Load failed...\n")

        # get file list of testing dataset
        import os, sys, inspect
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(currentdir)
        sys.path.insert(0, parentdir)
        from datasets import shanghaitech

        frames = shanghaitech.load_all_frames('../data/ShanghaiTech/part_A_final/test_data', load_labeling=False)
        img_clec, dmap_clec = load_data_pairs_v2(frames)


        # img_list = glob('{}/*.jpg'.format(self.testImagePath))
        # img_list.sort()
        # dmap_list = glob('{}/*.mat'.format(self.testDmapPath))
        # dmap_list.sort()
        # img_clec, dmap_clec = load_data_pairs(img_list, dmap_list)

        # self.test_training(img_clec, dmap_clec, 0, log_file)

        for i_dx in xrange(len(img_clec[0:6])):
            # train batch
            print_i = '{:05d}'.format(i_dx + 1)
            img_data = img_clec[i_dx]
            print(img_data.shape)

            w, h, c = img_data.shape
            w = int(w / 4) * 4
            h = int(h / 4) * 4

            misc.imsave('output/u_i_{}.png'.format(print_i), img_data)

            img_data = resize(img_data, (w, h, c), preserve_range=True)
            img_data = img_data.reshape(1, w, h, c)

            tbegin = datetime.datetime.now()
            predicted_label = self.sess.run(self.pred_prob, feed_dict={self.input_Img: img_data})
            print("Running model: {}ms".format(int((datetime.datetime.now() - tbegin).total_seconds() * 1000)))
            predicted_label /= 100.0
            predicted_label = np.squeeze(predicted_label)
            predicted_label = predicted_label/predicted_label.max() * 255.
            misc.imsave('output/u_r_{}.png'.format(print_i), predicted_label)

        print("DONE!!!")

    def inference(self, img_path, dmap_path):

        print("Starting test Process:\n")

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        img_data = ReadImage(img_path)
        img_data = img_data / 255.0

        dmap_data = get_dmap(img_path, dmap_path)

        w, h, c = img_data.shape
        w = int(w / 4) * 4
        h = int(h / 4) * 4

        img_data = resize(img_data, (w, h, c), preserve_range=True)
        img_data = img_data.reshape(1, w, h, c)

        predicted_label = self.sess.run(extract_keypoints(self.pred_prob), feed_dict={self.input_Img: img_data})

        # print np.max(predicted_label)
        # predicted_label = predicted_label/100.0
        print
        np.sum(predicted_label)
        print
        np.sum(dmap_data)

        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(img_data[0])
        fig.add_subplot(1, 3, 2)
        plt.imshow(dmap_data, cmap=plt.cm.gray)
        fig.add_subplot(1, 3, 3)
        plt.imshow(predicted_label[0, :, :, 0], cmap=plt.cm.gray)
        plt.show()

    def visualize(self):

        print("Starting test Process:\n")

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # print [v.name for v in tf.all_variables()]
        with tf.variable_scope('RBF_gt', reuse=True) as scope:
            w1_tf = tf.get_variable('w1')
            w1 = self.sess.run(w1_tf)
            print
            w1

            w2_tf = tf.get_variable('w2')
            w2 = self.sess.run(w2_tf)
            print
            w2

            w3_tf = tf.get_variable('w3')
            w3 = self.sess.run(w3_tf)
            print
            w3

            w4_tf = tf.get_variable('w4')
            w4 = self.sess.run(w4_tf)
            print
            w4

            w5_tf = tf.get_variable('w5')
            w5 = self.sess.run(w5_tf)
            print
            w5

    # save checkpoint file
    def save_chkpoint(self, checkpoint_dir, model_name, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    # load checkpoint file
    def load_chkpoint(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        checkpoint_dir = os.path.join(checkpoint_dir, self.load_model_path)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
