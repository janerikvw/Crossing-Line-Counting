import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

import getopt
import math
import numpy
import os
import time
import PIL
import PIL.Image
import sys
import argparse
from matplotlib import pyplot as plt
import cv2
from datetime import datetime

import model
import utils
import model_utils

from dataset import SimpleDataset


# Add base path to import dir for importing datasets
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from datasets import tub

parser = argparse.ArgumentParser(description='PyTorch DDFlow (PWCNet backbone)')

parser.add_argument('results_dir',metavar='RESULTS', type=str,
                    help='Directory where to store all the results')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')


def forward_step(frame1, frame2, args, net):
    int_width = frame1.shape[3]
    int_height = frame1.shape[2]
    int_preprocessed_width = int(math.floor(math.ceil(int_width / 64.0) * 64.0))
    int_preprocessed_height = int(math.floor(math.ceil(int_height / 64.0) * 64.0))

    frame1 = torch.nn.functional.interpolate(input=frame1,
                                             size=(int_preprocessed_height, int_preprocessed_width),
                                             mode='bilinear', align_corners=False)
    frame2 = torch.nn.functional.interpolate(input=frame2,
                                             size=(int_preprocessed_height, int_preprocessed_width),
                                             mode='bilinear', align_corners=False)

    # Feed through encoder and decode forward and backward
    features1 = net.netExtractor(frame1)
    features2 = net.netExtractor(frame2)
    flow_fw = net.decode(features1, features2)
    flow_bw = net.decode(features2, features1)

    multiplication = 20.0 # (int_height / flow_fw.shape[2]) * (int_width / flow_fw.shape[3])

    flow_fw = multiplication * torch.nn.functional.interpolate(input=flow_fw, size=(int_height, int_width),
                                                               mode='bilinear', align_corners=False)

    flow_bw = multiplication * torch.nn.functional.interpolate(input=flow_bw, size=(int_height, int_width),
                                                               mode='bilinear', align_corners=False)

    flow_fw[:, 0, :, :] *= float(int_width) / float(int_preprocessed_width)
    flow_fw[:, 1, :, :] *= float(int_height) / float(int_preprocessed_height)

    flow_bw[:, 0, :, :] *= float(int_width) / float(int_preprocessed_width)
    flow_bw[:, 1, :, :] *= float(int_height) / float(int_preprocessed_height)

    return flow_fw, flow_bw

def abs_robust_loss(diff, mask, q=0.4):
    diff = torch.pow(torch.abs(diff)+0.01, q)
    diff = torch.mul(diff, mask)
    diff_sum = diff.sum()
    loss_mean = diff_sum / (mask.sum() * 2 + 1e-6)
    return loss_mean

def calc_loss(frame1, frame2, flow_fw, flow_bw):
    occ_fw, occ_bw = model_utils.occlusion(flow_fw, flow_bw)
    mask_fw = 1. - occ_fw
    mask_bw = 1. - occ_bw

    img1_warp = model_utils.backwarp(frame1, flow_bw)
    img2_warp = model_utils.backwarp(frame2, flow_fw)

    # Calc photometric loss
    loss1 = abs_robust_loss(frame1 - img2_warp, torch.ones_like(mask_fw))
    loss2 = abs_robust_loss(frame2 - img1_warp, torch.ones_like(mask_bw))
    photometric_loss = loss1 + loss2
    return photometric_loss

# Perform a simple test and save
def test(net, dataset, args, iter):
    str_i = '{0:06d}'.format(iter + 1)

    net.eval()

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=args.dataloader_workers,
                            pin_memory=True)

    for o, (frame1, frame2) in enumerate(dataloader):
        # For now just the first one, later random sampling (So different dataloader)
        if o != 0:
            break

        str_o = '{0:03d}'.format(o + 1)
        frame1 = frame1.cuda()
        frame2 = frame2.cuda()
        flow_fw, flow_bw = forward_step(frame1, frame2, args, net)
        occ_fw, occ_bw = model_utils.occlusion(flow_fw, flow_bw)

        flow_fw = flow_fw.detach().cpu().numpy().transpose(0, 2, 3, 1)
        rgb = utils.flo_to_color(flow_fw[0])
        plt.imsave('results/{}/{}_{}_flow_fw.png'.format(args.save_dir, str_i, str_o), rgb)

        frame1_save = frame1.cpu().numpy().transpose(0, 2, 3, 1)
        plt.imsave('results/{}/{}_{}_orig.png'.format(args.save_dir, str_i, str_o), frame1_save[0])

        occ_fw_save = occ_fw.detach().cpu().numpy().transpose(0, 2, 3, 1)
        plt.imsave('results/{}/{}_{}_occ_fw.png'.format(args.save_dir, str_i, str_o), occ_fw_save[0].squeeze())

    net.train()


def train(args):
    torch.cuda.manual_seed(args.seed)

    # Load model
    net = model.PWCNet()
    if args.pre:
        net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                    torch.load(args.pre).items()})
    net = net.cuda()
    net.train()


    # Get dataset and dataloader
    train_pairs = []
    for video in tub.load_all_videos('../data/TUBCrowdFlow', load_peds=False):
        train_video, _, val_video, _, _, _ = tub.train_val_test_split(video, None)
        train_pairs += train_video.get_frame_pairs()

    dataset = SimpleDataset(train_pairs)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            sampler=RandomSampler(dataset, replacement=True, num_samples=args.iters*args.batch_size),
                            num_workers=args.dataloader_workers,
                            pin_memory=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.init_lr, weight_decay=args.regularization_weight)
    optim_lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)

    # Start tensorboard writer
    writer = SummaryWriter(log_dir='summaries/{}'.format(args.save_dir))
    os.mkdir('results/{}/'.format(args.save_dir))
    os.mkdir('weights/{}/'.format(args.save_dir))

    print("Start training")

    timer = utils.sTimer('Full update')

    for i, (frame1, frame2) in enumerate(dataloader):

        str_i = '{0:06d}'.format(i+1)
        frame1 = frame1.cuda()
        frame2 = frame2.cuda()

        flow_fw, flow_bw = forward_step(frame1, frame2, args, net)
        loss = calc_loss(frame1, frame2, flow_fw, flow_bw)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), i)


        # Print results every and do a speed test
        if i == 0 or (i + 1) % args.print_every == 0:
            print('[{}/{}] - loss({})'.format(str_i, args.iters, loss.item()))
            test(net, dataset, args, i)

        # Save results every
        if (i + 1) % args.save_every == 0:
            torch.save(net.state_dict(), 'weights/{}/model_{}.pt'.format(args.save_dir, str_i))

        # Decay learning rate
        if (i+1)%args.lr_decay_every == 0:
            optim_lr_decay.step()

        writer.add_scalar('Time/full_iter', timer.show(False), i)
        timer = utils.sTimer('Full update')

    writer.close()
    return

if __name__ == '__main__':
    args = parser.parse_args()
    args.batch_size = 1
    args.iters = 200000 #10

    # Keep these fixed to make sure reproducibility
    args.dataloader_workers = 1
    args.seed = 127 # time.time()

    args.regularization_weight = 0 # 1e-4 * 4
    args.init_lr = 5e-5
    args.lr_decay_rate = 0.5
    args.lr_decay_every = 50000

    args.print_every = 200
    args.save_every = 2000

    args.save_dir = '{}_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"), args.results_dir)

    train(args)