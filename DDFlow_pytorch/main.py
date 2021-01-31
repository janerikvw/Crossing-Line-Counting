import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

import getopt
import random
import math
import numpy
import os
import time
import PIL
import PIL.Image
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

parser.add_argument('--data_path', '-d', metavar='DATA_PATH', default='../data/TUBCrowdFlow',type=str,
                    help='Path to the TUB dataset')

# Load the FramePairs into a SimpleDataset format.
# Makes it pretty easy to swap datasets
def load_dataset(args, between=1, load_gt=False):
    # Get dataset and dataloader
    train_pairs = []
    val_pairs = []
    for video in tub.load_all_videos(args.data_path, load_peds=False):
        train_video, _, val_video, _, _, _ = tub.train_val_test_split(video, None)
        train_video.generate_frame_pairs(between)
        train_pairs += train_video.get_frame_pairs()
        
        val_video.generate_frame_pairs(between)
        val_pairs += val_video.get_frame_pairs()

    return SimpleDataset(train_pairs, load_gt), SimpleDataset(val_pairs)

# Load the PWCNet model
def load_model(args):
    # Load model
    net = model.PWCNet()
    if args.pre:
        # Exception with old model. Easy to transform into original weight
        if args.pre == 'network-chairs-things.pytorch':
            net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                 torch.load(args.pre).items()})
        else:
            net.load_state_dict(torch.load(args.pre))
    net = net.cuda()
    return net


# Perform a simple test and save for each prediction the original, flow map and occlusion map
def simple_test(net, dataset, args, iter, count=-1):
    str_i = '{0:06d}'.format(iter + 1)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=args.dataloader_workers,
                            pin_memory=True)

    for o, (frame1, frame2) in enumerate(dataloader):
        # For now just the first one, later random sampling (So different dataloader)
        if o == count and -1 != count:
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

        if (o+1)%args.test_print_every == 0 and -1 != args.test_print_every:
            print('[{0:04d}/{1:04d}]'.format(o + 1, len(dataset)))


def train(args):
    # Load model
    net = load_model(args)
    net.train()

    # Get dataset and dataloader
    train_dataset, val_dataset = load_dataset(args)
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
        photo_losses = losses.create_photometric_losses(frame1, frame2, flow_fw, flow_bw)
        loss = photo_losses['abs_robust_mean']['non_occlusion']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Add to tensorboard
        writer.add_scalar('Loss/train', loss.item(), i)


        # Print results every and do a speed test
        if i == 0 or (i + 1) % args.print_every == 0:
            print('[{}/{}] - loss({})'.format(str_i, args.iters, loss.item()))
            # Now the training dataset is used for the test, but in real life we should apply the validation set
            net.eval()
            simple_test(net, val_dataset, args, i, 1)
            net.train()

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

def patch_train(args):
    # Load model
    net = load_model(args)
    net.train()

    # Get dataset and dataloader
    train_dataset, val_dataset = load_dataset(args, load_gt=args.prefix_gen)
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

    for i, (c_frame1, c_frame2, c_flow_fw, c_flow_bw, _, _) in enumerate(dataloader):
        str_i = '{0:06d}'.format(i+1)

        c_frame1, c_frame2, c_flow_fw, c_flow_bw = c_frame1.cuda(), c_frame2.cuda(),\
                                                   c_flow_fw.cuda(), c_flow_bw.cuda()

        # Select patch
        start_c_x = random.randrange(args.patch_off_border, c_frame1.shape[3]-args.patch_target_w-args.patch_off_border)
        start_c_y = random.randrange(args.patch_off_border, c_frame1.shape[2]-args.patch_target_h-args.patch_off_border)
        end_c_x = start_c_x + args.patch_target_w
        end_c_y = start_c_y + args.patch_target_h

        # Crop all the results
        c_frame1, c_frame2,\
        c_flow_fw, c_flow_bw = c_frame1[:,:,start_c_y:end_c_y,start_c_x:end_c_x],\
                               c_frame2[:,:,start_c_y:end_c_y,start_c_x:end_c_x],\
                               c_flow_fw[:,:,start_c_y:end_c_y,start_c_x:end_c_x],\
                               c_flow_bw[:,:,start_c_y:end_c_y,start_c_x:end_c_x]


        # Extract flow forward and backward (frame2->frame2) and calculate the loss
        flow_fw, flow_bw = net.bidirection_forward(c_frame1, c_frame2)

        # Currently only using non_occ_photometric_loss
        # @TODO Make more clear when changing between losses for different stages
        photo_losses = losses.create_photometric_losses(c_frame1, c_frame2, flow_fw, flow_bw)
        patch_loss = losses.create_distilled_losses(flow_fw, flow_bw, c_flow_fw, c_flow_bw)
        loss = photo_losses['abs_robust_mean']['occlusion'] + patch_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Add to tensorboard
        writer.add_scalar('Loss/train', loss.item(), i)


        # Print results every and do a speed test
        if i == 0 or (i + 1) % args.print_every == 0:
            print('[{}/{}] - loss({})'.format(str_i, args.iters, loss.item()))
            # Now the training dataset is used for the test, but in real life we should apply the validation set
            net.eval()
            simple_test(net, val_dataset, args, i, 1)
            net.train()

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


def test(args):
    # Load model
    net = load_model(args)
    net.eval()

    print("Images will be saved in: {}".format(args.save_dir))
    Path('test_results/{}/'.format(args.name)).mkdir(parents=True, exist_ok=True)

    # Get dataset and dataloader
    _, val_dataset = load_dataset(args)
    simple_test(net, val_dataset, args, 0)


def generate(args):
    if not args.prefix_gen:
        print("Prefix for generation can't be None")
        return

    # Load model
    net = load_model(args)
    net.eval()

    train_dataset, _ = load_dataset(args)

    dataloader = DataLoader(train_dataset,
                            batch_size=1,
                            num_workers=args.dataloader_workers,
                            pin_memory=True)

    for o, (frame1, frame2) in enumerate(dataloader):
        pair = train_dataset.pairs[o]
        print('{0:05d}/{1:05d} - {2}'.format(o+1, len(train_dataset.pairs), pair.get_frames(0).get_info_dir()))
        frame1 = frame1.cuda()
        frame2 = frame2.cuda()
        flow_fw, flow_bw = net.bidirection_forward(frame1, frame2)
        occ_fw, occ_bw = model_utils.occlusion(flow_fw, flow_bw)

        utils.save_gt(pair, flow_fw.detach().cpu()[0],
                      flow_bw.detach().cpu()[0], occ_fw.detach().cpu()[0],
                      occ_bw.detach().cpu()[0], args.prefix_gen)


if __name__ == '__main__':
    args = parser.parse_args()
    args.batch_size = 1
    args.iters = 150000 #10

    # Keep these fixed to make sure reproducibility
    args.dataloader_workers = 1
    args.seed = 127 # time.time()

    args.regularization_weight = 0 # 1e-4 * 4
    args.init_lr = 5e-5
    args.lr_decay_rate = 0.5
    args.lr_decay_every = 50000

    args.print_every = 500
    args.save_every = 5000

    args.prefix_gen = 'v1_'

    args.patch_off_border = 64
    args.patch_target_h = 256
    args.patch_target_w = 640

    if args.mode == 'test':
        args.test_print_every = 10
    else:
        args.test_print_every = -1


    # Add date and time so we can just run everything very often :)
    args.save_dir = '{}_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"), args.name)

    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    if args.mode == 'test':
        test(args)
    elif args.mode == 'generate':
        generate(args)
    elif args.mode == 'train2':
        patch_train(args)
    elif args.mode == 'train1':
        train(args)