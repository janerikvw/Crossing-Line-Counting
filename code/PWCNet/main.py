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
from pathlib import Path

import model
import utils
import model_utils
import losses

from dataset import SimpleDataset


# Add base path to import dir for importing datasets
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from datasets import tub

parser = argparse.ArgumentParser(description='PyTorch DDFlow (PWCNet backbone)')

parser.add_argument('name',metavar='NAME', type=str,
                    help='Used as postfix for the save directory')

parser.add_argument('--mode', '-m', metavar='MODUS', default='train', type=str,
                    help='Which modus we use? train,test,generate')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default='network-chairs-things.pytorch',type=str,
                    help='path to the pretrained model')

# Load the FramePairs into a SimpleDataset format.
# Makes it pretty easy to swap datasets
def load_dataset():
    # Get dataset and dataloader
    train_pairs = []
    val_pairs = []
    for video in tub.load_all_videos('../data/TUBCrowdFlow', load_peds=False):
        train_video, _, val_video, _, _, _ = tub.train_val_test_split(video, None)
        train_pairs += train_video.get_frame_pairs()
        val_pairs += val_video.get_frame_pairs()

    train_dataset = SimpleDataset(train_pairs)
    val_dataset = SimpleDataset(train_pairs)

    return train_dataset, val_dataset

# Perform a simple test and save for each prediction the original, flow map and occlusion map
def simple_test(net, dataset, args, iter):
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
        flow_fw, flow_bw = net.bidirection_forward(frame1, frame2)
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
    train_dataset, val_dataset = load_dataset()
    dataloader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            sampler=RandomSampler(train_dataset, replacement=True, num_samples=args.iters*args.batch_size),
                            num_workers=args.dataloader_workers,
                            pin_memory=True)

    # Init optimizer and lr decay class
    optimizer = torch.optim.Adam(net.parameters(), lr=args.init_lr, weight_decay=args.regularization_weight)
    optim_lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)

    # Start tensorboard writer and init results directories
    writer = SummaryWriter(log_dir='summaries/{}'.format(args.save_dir))
    Path('results/{}/'.format(args.save_dir)).mkdir(parents=True, exist_ok=True)
    Path('weights/{}/'.format(args.save_dir)).mkdir(parents=True, exist_ok=True)

    # Timer
    timer = utils.sTimer('Full update')

    for i, (frame1, frame2) in enumerate(dataloader):

        str_i = '{0:06d}'.format(i+1)
        frame1 = frame1.cuda()
        frame2 = frame2.cuda()

        # Extract flow forward and backward (frame2->frame2) and calculate the loss
        flow_fw, flow_bw = net.bidirection_forward(frame1, frame2)
        # Currently only using non_occ_photometric_loss
        # @TODO Make more clear when changing between losses for different stages
        loss, _ = losses.create_photometric_losses(frame1, frame2, flow_fw, flow_bw)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Add to tensorboard
        writer.add_scalar('Loss/train', loss.item(), i)


        # Print results every and do a speed test
        if i < 3 or (i + 1) % args.print_every == 0:
            print('[{}/{}] - loss({})'.format(str_i, args.iters, loss.item()))
            # Now the training dataset is used for the test, but in real life we should apply the validation set
            simple_test(net, val_dataset, args, i)

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

    # Add date and time so we can just run everything very often :)
    args.save_dir = '{}_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"), args.name)

    train(args)