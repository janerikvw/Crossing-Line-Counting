import argparse
import time
import math
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from glob import glob
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

# from model import DRNetModel
from dataset import SimpleDataset
from models import V3EndFlow, \
    V3Dilation, V32Dilation, V3EndFlowDilation, V32EndFlowDilation,\
    V33Dilation, V332SingleFlow,\
    V33EndFlowDilation, V34EndFlowDilation, V35EndFlowDilation, V332EndFlowDilation,\
    V332Dilation, V333Dilation, V34Dilation, V341Dilation, V35Dilation, V351Dilation, V3Correlation, Baseline1,\
    V5Dilation, V501Dilation, V51Dilation, V52Dilation, V5Flow, V5FlowFeatures, V51FlowFeatures, V5FlowWarping, V51FlowWarping,\
    Baseline2, Baseline21, V6Blocker, V61Blocker, V62Blocker, V601Blocker, V55FlowWarping
from dense_models import P1Base, P2Base
from PIL import Image, ImageDraw
from tqdm import tqdm

from pathlib import Path
import utils
import loi
import density_filter

# Add base path to import dir for importing datasets
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from datasets import basic_entities, fudan, ucsdpeds, dam, aicity, tub
from DDFlow_pytorch import losses
from CSRNet.model import CSRNet

print("Model training")
print("Input: {}".format(str(sys.argv)))

parser = argparse.ArgumentParser(description='CrowdCounting (PWCNet backbone)')

parser.add_argument('name', metavar='NAME', type=str,
                    help='Used as postfix for the save directory')

parser.add_argument('--mode', '-m', metavar='MODE', type=str, default='train',
                    help='Train or test')

parser.add_argument('--pre', metavar='PRETRAINED_MODEL', default='', type=str,
                    help='Path to the TUB model')

parser.add_argument('--dataset', metavar='DATSET', default='fudan', type=str,
                    help='Selected dataset (fudan/tub)')

parser.add_argument('--density_model', metavar='DENSITY', default='fixed-8', type=str,
                    help='Selected density map: fixed-* (3,5,8,12,16)')

parser.add_argument('--loss_focus', metavar='LOSS_FOCUS', default='full', type=str,
                    help='Which loss to use (full, fe, cc)')

parser.add_argument('--cc_weight', metavar='CC_WEIGHT', default=5, type=float,
                    help='Weight of cc in comparing to fe for full loss')

parser.add_argument('--frames_between', metavar='SKIP_FRAMES', default=5, type=int,
                    help='Skipping frames, most of datasets have fps of 25, so skipping 5 makes 5fps')

parser.add_argument('--epochs', metavar='EPOCHS', default=500, type=int,
                    help='Epochs running')

parser.add_argument('--student_model', metavar='STUDENT', default='off', type=str,
                    help='Run the student model for optimal flow halfway')

parser.add_argument('--model', metavar='MODEL', default='v332singleflow', type=str,
                    help='Which model gonna train')

parser.add_argument('--resize_patch', metavar='RESIZE_PATCH', default='off', type=str,
                    help='Resizing patch so zoom/not zoom')

parser.add_argument('--resize_mode', metavar='RESIZE_MODE', default='bilinear', type=str,
                    help='Resizing patch so zoom/not zoom')

parser.add_argument('--lr_setting', metavar='LR_SETTING', default='adam_2', type=str,
                    help='Give a specific learning rate setting')

parser.add_argument('--loi_maxing', metavar='LOI_REALIGNING', default=0, type=int,
                    help='Using maxing for LOI or not')

parser.add_argument('--loi_version', metavar='LOI_VERSION', default='v2', type=str,
                    help='Versio which is used to select lines')

parser.add_argument('--loi_width', metavar='LOI_WIDTH', default=40, type=int,
                    help='Width of the LOI')

parser.add_argument('--loi_level', metavar='LOI_LEVEL', default='pixel', type=str,
                    help='Method to merge the density and flow, region level or pixel level')

parser.add_argument('--loi_height', metavar='LOI_HEIGHT', default=2.0, type=float,
                    help='How many times the width to get good results')


def save_sample(args, dir, info, density, true, img, flow):
    save_image(img, '{}/{}/img.png'.format(dir, args.save_dir))
    save_image(true / true.max(), '{}/{}/true.png'.format(dir, args.save_dir))
    # save_image(density / density.max(), '{}/{}/pred.png'.format(dir, args.save_dir))
    # plt.imsave('{}/{}/flow.png'.format(dir, args.save_dir), flow)
    save_image(density / density.max(), '{}/{}/pred_{}.png'.format(dir, args.save_dir, info))
    if flow is not None:
        plt.imsave('{}/{}/flow_{}.png'.format(dir, args.save_dir, info), flow)


def n_split_pairs(frames, splits=3, distance=20, skip_inbetween=False):
    ret = []
    for s_split in np.array_split(frames, splits):
        ret.append(basic_entities.generate_frame_pairs(s_split, distance, skip_inbetween))

    return ret


def load_train_dataset(args):
    splits = [[] for _ in range(args.train_split)]
    total_num = 0

    if args.dataset == 'fudan':
        for video_path in glob('../data/Fudan/train_data/*/'):
            video = fudan.load_video(video_path)
            total_num += len(video.get_frames())
            splitted_frames = n_split_pairs(video.get_frames(), args.train_split, args.frames_between, skip_inbetween=False)

            # Possibly overfit slightly, because middle parts could almost overlap with outer parts, so shuffle to balance
            random.shuffle(splitted_frames)
            for i, split in enumerate(splitted_frames):
                splits[i] += split

    elif args.dataset == 'ucsd':
        videos = ucsdpeds.load_videos('../data/ucsdpeds')
        videos = videos[3:7] + videos[10:]
        for video in videos:
            total_num += len(video.get_frames())
            splitted_frames = n_split_pairs(video.get_frames(), args.train_split, args.frames_between,
                                            skip_inbetween=False)

            # Possibly overfit slightly, because middle parts could almost overlap with outer parts, so shuffle to balance
            random.shuffle(splitted_frames)
            for i, split in enumerate(splitted_frames):
                splits[i] += split

    elif args.dataset == 'dam':
        pairs = dam.load_all_pairs('../data/Dam', distance=args.frames_between)
        total_num += len(pairs)
        random.shuffle(pairs)
        for i, v in enumerate(pairs):
            splits[i % args.train_split].append(v)
    elif args.dataset == 'aicity':
        train_videos, _ = aicity.split_train_test(aicity.load_all_videos('../data/AICity'))
        for video in train_videos:
            total_num += len(video.get_frames())
            splitted_frames = n_split_pairs(video.get_frames(), args.train_split, args.frames_between,
                                            skip_inbetween=False)

            random.shuffle(splitted_frames)
            for i, split in enumerate(splitted_frames):
                splits[i] += split

    elif args.dataset == 'tub':
        train_videos, _, train_videos2 = tub.train_val_test_split(tub.load_all_videos('../data/TUBCrowdFlow'), 0.1, 0.1)
        for video in train_videos+train_videos2:
            total_num += len(video.get_frames())
            splitted_frames = n_split_pairs(video.get_frames(), args.train_split, args.frames_between,
                                            skip_inbetween=False)

            random.shuffle(splitted_frames)
            for i, split in enumerate(splitted_frames):
                splits[i] += split
    else:
        print("No valid dataset selected!!!!")
        exit()

    print("Total frames loaded:", total_num)
    return splits


def setup_train_cross_dataset(splits, epoch, args):
    test_th = epoch % len(splits)
    train_pairs = []
    cross_pairs = splits[test_th]
    for i, split in enumerate(splits):
        if i == test_th:
            continue
        train_pairs += split

    if len(train_pairs) > args.train_amount:
        train_pairs = random.sample(train_pairs, args.train_amount)

    if len(cross_pairs) > args.cross_val_amount:
        cross_pairs = random.sample(cross_pairs, args.cross_val_amount)

    return (SimpleDataset(train_pairs, args, True),
            SimpleDataset(cross_pairs, args, False))


def load_test_dataset(args):
    test_vids = []
    cross_vids = []

    if args.dataset == 'fudan':
        for video_path in glob('../data/Fudan/test_data/*/'):
            test_vids.append(fudan.load_video(video_path))
            cross_vids.append(fudan.load_video(video_path))
    elif args.dataset == 'ucsd':
        videos = ucsdpeds.load_videos('../data/ucsdpeds')
        # Cross Improve!!
        cross_vids = videos[:3] + videos[7:10]
        test_vids = videos[:3] + videos[7:10]
    elif args.dataset == 'dam':
        # Single test video
        video = dam.load_test_video('../data/Dam/test_arena2')
        cross_vids = [video]
        test_vids = [video]
    elif args.dataset == 'aicity':
        _, test_videos = aicity.split_train_test(aicity.load_all_videos('../data/AICity'), train=0.5)
        #cross_vids, test_vids = aicity.split_train_test(test_videos, train=0.5)
        cross_vids = test_videos
        test_vids = test_videos
    elif args.dataset == 'tub':
        _, rest, _ = tub.train_val_test_split(
            tub.load_all_videos('../data/TUBCrowdFlow'), 0.1, 0.1)
        cross_vids = rest
        test_vids = rest
        # cross_vids, test_vids = tub.split_train_test(rest, 0.5)
    else:
        print("No valid dataset selected!!!!")
        exit()

    return cross_vids, test_vids



def load_model(args):
    model = None
    if args.model == 'p1base':
        model = P1Base(load_pretrained=True).cuda()
    elif args.model == 'p2base':
        model = P2Base(load_pretrained=True).cuda()
    elif args.model == 'old_v31':
        model = ModelV31(load_pretrained=True).cuda()
    elif args.model == 'v3dilation':
        model = V3Dilation(load_pretrained=True).cuda()
    elif args.model == 'v32dilation':
        model = V32Dilation(load_pretrained=True).cuda()
    elif args.model == 'v33dilation':
        model = V33Dilation(load_pretrained=True).cuda()
    elif args.model == 'v332dilation':
        model = V332Dilation(load_pretrained=True).cuda()
    elif args.model == 'v332singleflow':
        model = V332SingleFlow(load_pretrained=True).cuda()
    elif args.model == 'v333dilation':
        model = V333Dilation(load_pretrained=True).cuda()
    elif args.model == 'v34dilation':
        model = V34Dilation(load_pretrained=True).cuda()
    elif args.model == 'v35dilation':
        model = V35Dilation(load_pretrained=True).cuda()
    elif args.model == 'v351dilation':
        model = V351Dilation(load_pretrained=True).cuda()
    elif args.model == 'v341dilation':
        model = V341Dilation(load_pretrained=True).cuda()
    elif args.model == 'v5dilation':
        model = V5Dilation(load_pretrained=True).cuda()
    elif args.model == 'v501dilation':
        model = V501Dilation(load_pretrained=True).cuda()
    elif args.model == 'v51dilation':
        model = V51Dilation(load_pretrained=True).cuda()
    elif args.model == 'v52dilation':
        model = V52Dilation(load_pretrained=True).cuda()
    elif args.model == 'v5flow':
        model = V5Flow(load_pretrained=True).cuda()
    elif args.model == 'v6blocker':
        model = V6Blocker(load_pretrained=True).cuda()
    elif args.model == 'v601blocker':
        model = V601Blocker(load_pretrained=True).cuda()
    elif args.model == 'v61blocker':
        model = V61Blocker(load_pretrained=True).cuda()
    elif args.model == 'v62blocker':
        model = V62Blocker(load_pretrained=True).cuda()
    elif args.model == 'v5flowfeatures':
        model = V5FlowFeatures(load_pretrained=True).cuda()
    elif args.model == 'v51flowfeatures':
        model = V51FlowFeatures(load_pretrained=True).cuda()
    elif args.model == 'v5flowwarping':
        model = V5FlowWarping(load_pretrained=True).cuda()
    elif args.model == 'v55flowwarping':
        model = V55FlowWarping(load_pretrained=True).cuda()
    elif args.model == 'v51flowwarping':
        model = V51FlowWarping(load_pretrained=True).cuda()
    elif args.model == 'v3endflowdilation':
        model = V3EndFlowDilation(load_pretrained=True).cuda()
    elif args.model == 'v32endflowdilation':
        model = V32EndFlowDilation(load_pretrained=True).cuda()
    elif args.model == 'v33endflowdilation':
        model = V33EndFlowDilation(load_pretrained=True).cuda()
    elif args.model == 'v332endflowdilation':
        model = V332EndFlowDilation(load_pretrained=True).cuda()
    elif args.model == 'v34endflowdilation':
        model = V34EndFlowDilation(load_pretrained=True).cuda()
    elif args.model == 'v35endflowdilation':
        model = V35EndFlowDilation(load_pretrained=True).cuda()
    elif args.model == 'baseline1':
        model = Baseline1(load_pretrained=True).cuda()
    elif args.model == 'baseline2':
        model = Baseline2(load_pretrained=True).cuda()
    elif args.model == 'baseline21':
        model = Baseline21(load_pretrained=True).cuda()
    elif args.model == 'csrnet':
        model = CSRNet().cuda()
        args.loss_focus = 'cc'
    elif args.model == 'v3endflow':
        model = V3EndFlow(load_pretrained=True).cuda()
    else:
        print("Error! Incorrect model selected")
        exit()

    if args.pre:
        print("Load pretrained model:", args.pre)
        model.load_state_dict(torch.load(args.pre))

    model.train()

    return model


def train(args):
    print('Initializing result storage...')
    writer = SummaryWriter(log_dir='summaries/{}'.format(args.save_dir))
    Path('weights/{}/'.format(args.save_dir)).mkdir(parents=True, exist_ok=True)
    Path('results/{}/'.format(args.save_dir)).mkdir(parents=True, exist_ok=True)
    Path('finals/').mkdir(parents=True, exist_ok=True)

    print('Initializing dataset...')
    train_pair_splits = load_train_dataset(args)

    print('Initializing model...')
    model = load_model(args)

    if args.loss_function == 'L1':
        criterion = nn.L1Loss(reduction='sum').cuda()
    elif args.loss_function == 'L2':
        criterion = nn.MSELoss(reduction='sum').cuda()
    else:
        print("Error, no correct loss function")
        exit()

    if args.optimizer == 'adam':
        if args.lr_setting == 'adam_1':
            optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
        elif args.lr_setting == 'adam_2':
            optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-4)
        elif args.lr_setting == 'adam_5_no':
            optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0)
        elif args.lr_setting == 'adam_8':
            ret_params = []
            for name, params in model.named_parameters():
                if name.split('.')[0] == 'fe_net':
                    ret_params.append({'params': params, 'lr': 4e-5})
                else:
                    ret_params.append({'params': params, 'lr': 8e-4})

            optimizer = optim.Adam(ret_params, lr=2e-5, weight_decay=1e-4)
        elif args.lr_setting == 'adam_6_yes':
            optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(model.parameters(), 1e-6, momentum=0.95, weight_decay=5*1e-4)

    if args.lr_setting == 'adam_8':
        optim_lr_decay = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    else:
        optim_lr_decay = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    best_mae = None
    best_mse = None
    best_loss = None
    print('Start training...')

    _, test_dataset = setup_train_cross_dataset(train_pair_splits, 0, args)
    # if args.single_dataset:
    #     train_dataset, test_dataset = setup_train_cross_dataset(train_pair_splits, 0, args)
    #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    #                                                num_workers=args.dataloader_workers)

    for epoch in range(args.epochs):
        # if not args.single_dataset:
        #     train_dataset, test_dataset = setup_train_cross_dataset(train_pair_splits, 0, args)
        #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        #                                                num_workers=args.dataloader_workers)

        train_dataset, _ = setup_train_cross_dataset(train_pair_splits, 0, args)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.dataloader_workers)

        loss_container = utils.AverageContainer()
        for i, batch in enumerate(train_loader):

            frames1, frames2, densities = batch
            frames1 = frames1.cuda()
            frames2 = frames2.cuda()
            densities = densities.cuda()

            # Set grad to zero
            optimizer.zero_grad()

            # Run model and optimize
            if args.model == 'csrnet':
                pred_densities = model(frames1)
                flow_fw = None
                flow_bw = None
            else:
                flow_fw, flow_bw, pred_densities = model(frames1, frames2)

            # Resizing back to original sizes
            factor = (densities.shape[2] * densities.shape[3]) / (
                pred_densities.shape[2] * pred_densities.shape[3])

            if epoch == 0 and i == 0:
                print("Resize factor: {}".format(factor))

            pred_densities = F.interpolate(input=pred_densities,
                                           size=(densities.shape[2], densities.shape[3]),
                                           mode=args.resize_mode, align_corners=False) / factor

            if args.loss_focus != 'cc':
                photo_losses = losses.create_photometric_losses(frames1, frames2, flow_fw, flow_bw)
                fe_loss = photo_losses['abs_robust_mean']['no_occlusion']

                loss_container['abs_no_occlusion'].update(photo_losses['abs_robust_mean']['no_occlusion'].item())
                loss_container['abs_occlusion'].update(photo_losses['abs_robust_mean']['occlusion'].item())
                loss_container['census_no_occlusion'].update(photo_losses['census']['no_occlusion'].item())
                loss_container['census_occlusion'].update(photo_losses['census']['occlusion'].item())

            if args.loss_focus != 'fe':
                cc_loss = criterion(pred_densities, densities.repeat(1, pred_densities.shape[1], 1, 1))

            if args.loss_focus == 'cc':
                loss = cc_loss * args.cc_weight
            elif args.loss_focus == 'fe':
                loss = fe_loss
            else:
                loss = fe_loss + cc_loss * args.cc_weight

            loss.backward()
            optimizer.step()

            loss_container['total_loss'].update(loss.item())

            if args.loss_focus != 'fe':
                loss_container['cc_loss'].update(cc_loss.item())
            if args.loss_focus != 'cc':
                loss_container['fe_loss'].update(fe_loss.item())

        if epoch > 0:
            writer.add_scalar('Train/Total_loss', loss_container['total_loss'].avg, epoch)
            writer.add_scalar('Train/FE_loss', loss_container['fe_loss'].avg, epoch)
            writer.add_scalar('Train/CC_loss', loss_container['cc_loss'].avg, epoch)

            writer.add_scalar('FE_loss/abs_no_occlusion', loss_container['abs_no_occlusion'].avg, epoch)
            writer.add_scalar('FE_loss/abs_occlusion', loss_container['abs_occlusion'].avg, epoch)
            writer.add_scalar('FE_loss/census_no_occlusion', loss_container['census_no_occlusion'].avg, epoch)
            writer.add_scalar('FE_loss/census_occlusion', loss_container['census_occlusion'].avg, epoch)

        if epoch % args.test_epochs == args.test_epochs - 1:
            timer = utils.sTimer('Test run')
            avg, avg_sq, avg_loss = test_run(args, epoch, test_dataset, model)
            writer.add_scalar('Val/eval_time', timer.show(False), epoch)
            writer.add_scalar('Val/MAE', avg.avg, epoch)
            writer.add_scalar('Val/MSE', avg_sq.avg, epoch)
            writer.add_scalar('Val/Loss', avg_loss.avg, epoch)
            torch.save(model.state_dict(), 'weights/{}/last_model.pt'.format(args.save_dir))
            if best_loss is None or best_loss > avg_loss.avg:
                best_mae = avg.avg
                best_mse = avg_sq.avg
                best_loss = avg_loss.avg
                torch.save(model.state_dict(), 'weights/{}/best_model.pt'.format(args.save_dir))
                print("----- NEW BEST!! -----")

        # Learning decay update
        optim_lr_decay.step()

    results = {
        'cross_mae': best_mae,
        'cross_mse': best_mse
    }

    with open("finals/{}.txt".format(args.save_dir), "w") as outfile:
        json.dump(results, outfile)
    return


def test_run(args, epoch, test_dataset, model, save=True):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=args.dataloader_workers)

    avg = utils.AverageMeter()
    avg_sq = utils.AverageMeter()
    avg_loss = utils.AverageMeter()

    truth = utils.AverageMeter()
    pred = utils.AverageMeter()

    if args.loss_function == 'L1':
        criterion = nn.L1Loss(reduction='sum').cuda()
    elif args.loss_function == 'L2':
        criterion = nn.MSELoss(reduction='sum').cuda()
    else:
        print("Error, no correct loss function")
        exit()

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            frames1, frames2, densities = batch
            frames1 = frames1.cuda()
            frames2 = frames2.cuda()
            densities = densities.cuda()

            if args.model == 'csrnet':
                pred_densities = model(frames1)
                flow_fw = None
            else:
                flow_fw, flow_bw, pred_densities = model(frames1, frames2)
                flow_fw = flow_fw.detach()
                flow_bw.detach()

            pred_densities = pred_densities.detach()

            factor = (densities.shape[2] * densities.shape[3]) / (
                    pred_densities.shape[2] * pred_densities.shape[3])
            pred_densities = F.interpolate(input=pred_densities,
                                           size=(densities.shape[2], densities.shape[3]),
                                           mode=args.resize_mode, align_corners=False) / factor

            truth.update(densities.sum().item())
            pred.update(pred_densities.sum().item())

            avg.update(abs((pred_densities.sum() - densities.sum()).item()))
            avg_sq.update(torch.pow(pred_densities.sum() - densities.sum(), 2).item())
            avg_loss.update(criterion(pred_densities, densities))

            if i == 1 and save:
                if args.loss_focus != 'cc':
                    flow_fw = flow_fw.detach().cpu().numpy().transpose(0, 2, 3, 1)
                    rgb = utils.flo_to_color(flow_fw[0])
                else:
                    rgb = None

                save_sample(args, 'results', epoch, pred_densities[0], densities[0], frames1[0], rgb)

    print("--- TEST [MAE: {}, RMSE: {}, LOSS: {}]".format(avg.avg, math.pow(avg_sq.avg, 0.5), avg_loss.avg))
    model.train()

    return avg, avg_sq, avg_loss


#### Smooth the surroundings for Flow Estimation ####
# Due to the use of 2D conv to do one sample at the time which is easy
# Update could handle multiple frames at the time
#
# Surrounding: Pixels around each pixel to look for the max
# only_under removes the search in top side of the pixel
# Smaller_sides: When only_under the width search will be as wide as the height (surrounding+1)
def get_max_surrounding(data, surrounding=1, only_under=True, smaller_sides=True):
    kernel_size = surrounding * 2 + 1

    out_channels = np.eye(kernel_size * kernel_size)

    if only_under:
        out_channels = out_channels[surrounding * kernel_size:]
        if smaller_sides:
            for i in range(math.floor(surrounding/2)):
                out_channels[list(range(i, len(out_channels), kernel_size))] = False
                out_channels[list(range(kernel_size - i - 1, len(out_channels), kernel_size))] = False

    w = out_channels.reshape((out_channels.shape[0], 1, kernel_size, kernel_size))
    w = torch.tensor(w, dtype=torch.float).cuda()

    data = data.transpose(0, 1).cuda()
    patches = torch.nn.functional.conv2d(data, w, padding=(surrounding, surrounding))[:, :, :data.shape[2], :data.shape[3]]

    speed = torch.sqrt(torch.sum(torch.pow(patches, 2), axis=0))
    max_speeds = torch.argmax(speed, axis=0)
    flat_max_speeds = max_speeds.reshape(data.shape[2] * data.shape[3])

    y_axis = torch.arange(data.shape[2]).repeat_interleave(data.shape[3])
    x_axis = torch.arange(data.shape[3]).repeat(data.shape[2])

    output = patches[:, flat_max_speeds, y_axis, x_axis]
    output = output.reshape(1, 2, data.shape[2], data.shape[3])
    return output

def loi_test(args):
    metrics = utils.AverageContainer()

    # args.save_dir = '20201006_190409_dataset-fudan_model-csrnet_cc_weight-50_epochs-500_lr_setting-adam_2_resize_mode-bilinear'
    # args.save_dir = '20201013_163654_dataset-ucsd_model-csrnet_cc_weight-50_frames_between-2_epochs-750_loss_focus-cc_lr_setting-adam_2_resize_mode-bilinear'
    # args.save_dir = '20201103_091739_dataset-tub_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-200_loss_focus-cc_lr_setting-adam_2_pre'
    # args.save_dir = '20201031_001332_dataset-aicity_model-csrnet_density_model-fixed-16_cc_weight-50_frames_between-2_epochs-400_loss_focus-cc_lr_setting-adam_2_resize_mode-bilinear'
    # args.model = 'csrnet'
    # args.loss_focus = 'cc'

    # args.save_dir = '20201030_190754_dataset-fudan_model-v5flowwarping_cc_weight-50_frames_between-5_epochs-250_lr_setting-adam_2_resize_mode-bilinear'
    # args.model = 'v5flowwarping'

    # args.save_dir = '20201106_060510_dataset-fudan_model-baseline2_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-500_lr_setting-adam_2'
    # args.save_dir = '20201107_171204_dataset-aicity_model-baseline2_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre'
    # args.model = 'baseline2'

    # args.save_dir = '20201105_113924_dataset-fudan_model-v51flowwarping_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-500_lr_setting-adam_2'
    # args.model = 'v51flowwarping'

    # args.save_dir = '20201027_073844_dataset-fudan_model-v5dilation_cc_weight-50_epochs-500_lr_setting-adam_2_resize_mode-bilinear'
    # args.model = 'v5dilation'

    #args.save_dir = '20201025_035340_dataset-fudan_model-v51flowfeatures_cc_weight-50_epochs-500_lr_setting-adam_2_resize_mode-bilinear'
    # args.save_dir = '20201024_191136_dataset-fudan_model-v51flowfeatures_cc_weight-50_epochs-250_lr_setting-adam_2_resize_mode-bilinear'
    #args.model = 'v51flowfeatures'

    if args.model == 'csrnet':
        args.loss_focus = 'cc'

    if args.pre == '':
        args.pre = 'weights/{}/best_model.pt'.format(args.save_dir)
    model = load_model(args)
    model.eval()
    if args.loss_focus == 'cc':
        fe_model = V332Dilation(load_pretrained=True).cuda()
        if args.dataset == 'fudan':
            fe_model = V3Correlation(load_pretrained=True).cuda()
            pre_fe = '20200916_093740_dataset-fudan_frames_between-5_loss_focus-fe_resize_patch-off'
        elif args.dataset == 'ucsd':
            pre_fe = '20201013_193544_dataset-ucsd_model-v332dilation_cc_weight-50_frames_between-2_epochs-750_loss_focus-fe_lr_setting-adam_2_resize_mode-bilinear'
        elif args.dataset == 'tub':
            pre_fe = '20201107_001146_dataset-tub_model-v332dilation_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-250_loss_focus-fe_lr_setting-adam_2'
        elif args.dataset == 'aicity':
            pre_fe = '20201106_223451_dataset-aicity_model-v332dilation_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_loss_focus-fe_lr_setting-adam_2'
        else:
            print("This dataset doesnt have flow only results")
            exit()

        fe_model.load_state_dict(
            torch.load('weights/{}/last_model.pt'.format(pre_fe))
        )
        fe_model.eval()

    results = []

    ucsd_total_count = [[], []]

    with torch.no_grad():
        # Right now on cross validation
        _, test_vids = load_test_dataset(args)
        for v_i, video in enumerate(test_vids[0:1]):

            vid_result = []
            print("Video:", video.get_path())

            video.generate_frame_pairs(distance=args.frames_between, skip_inbetween=True)
            dataset = SimpleDataset(video.get_frame_pairs(), args, False)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.dataloader_workers)

            # ---- REMOVE WHEN DONE ---- #
            if args.loi_flow_width:
                print("Loading flow")
                frames1, frames2, _ = next(iter(dataloader))
                frames1 = frames1.cuda()
                frames2 = frames2.cuda()
                fe_output, _, _ = fe_model.forward(frames1, frames2)
                fe_speed = torch.norm(fe_output, dim=1)

                threshold = 1.0
                fe_speed_high2 = fe_speed > threshold
                avg_speed = fe_speed[fe_speed_high2].sum()/fe_speed_high2.sum()

                if args.loi_flow_smoothing:
                    loi_width = int((0.75 + avg_speed / 10 * 0.25) * args.loi_width)
                else:
                    loi_width = int((avg_speed / 9) * args.loi_width)
            else:
                loi_width = args.loi_width

            for l_i, line in enumerate(video.get_lines()):
                if line.get_crossed()[0] + line.get_crossed()[1] == 0:
                    continue

                image = video.get_frame(0).get_image()
                width, height = image.size
                image.close()
                point1, point2 = line.get_line()

                loi_model = loi.LOI_Calculator(point1, point2,
                                               img_width=width, img_height=height,
                                               crop_processing=False,
                                               loi_version=args.loi_version, loi_width=loi_width,
                                               loi_height=loi_width*args.loi_height)
                loi_model.create_regions()

                total_d1 = 0
                total_d2 = 0
                totals = [total_d1, total_d2]
                crosses = line.get_crossed()

                per_frame = [[], []]

                pbar = tqdm(total=len(video.get_frame_pairs()))

                metrics['timing'].reset()

                for s_i, batch in enumerate(dataloader):
                    # if s_i < 18:
                    #     pbar.update(1)
                    #     continue
                    torch.cuda.empty_cache()
                    timer = utils.sTimer("Full process time")

                    frame_pair = video.get_frame_pairs()[s_i]
                    print_i = '{:05d}'.format(s_i + 1)
                    frames1, frames2, densities = batch
                    frames1 = loi_model.reshape_image(frames1.cuda())
                    frames2 = loi_model.reshape_image(frames2.cuda())

                    # Expand for CC only (CSRNet to optimize for real world applications)
                    # @TODO optimize for speed to show result
                    if args.loss_focus == 'cc':
                        if args.model == 'csrnet':
                            cc_output = model(frames1)
                        else:
                            _, _, cc_output = model.forward(frames1, frames2)

                        fe_output, _, _ = fe_model.forward(frames1, frames2)
                    else:
                        fe_output, _, cc_output = model.forward(frames1, frames2)

                    metrics['mae'].update(abs((cc_output.sum() - densities.sum()).item()))
                    metrics['mse'].update(torch.pow(cc_output.sum() - densities.sum(), 2).item())

                    if args.loi_maxing == 1:
                        if args.dataset == 'fudan':
                            fe_output = get_max_surrounding(fe_output, surrounding=6, only_under=True,
                                                            smaller_sides=True)
                            fe_output = get_max_surrounding(fe_output, surrounding=6, only_under=True,
                                                            smaller_sides=True)
                        elif args.dataset == 'tub':
                            fe_output = get_max_surrounding(fe_output, surrounding=6, only_under=True,
                                                            smaller_sides=True)
                        elif args.dataset == 'ucsd':
                            fe_output = get_max_surrounding(fe_output, surrounding=6, only_under=True,
                                                            smaller_sides=True)
                        elif args.dataset == 'aicity':
                            fe_output = get_max_surrounding(fe_output, surrounding=6, only_under=False,
                                                            smaller_sides=False)
                            fe_output = get_max_surrounding(fe_output, surrounding=6, only_under=False,
                                                            smaller_sides=False)
                        else:
                            print("No maxing exists for this dataset")
                            exit()

                    # Resize and save as numpy
                    cc_output = loi_model.to_orig_size(cc_output)
                    cc_output = cc_output.squeeze().squeeze()
                    cc_output = cc_output.detach().cpu().data.numpy()
                    fe_output = loi_model.to_orig_size(fe_output)
                    fe_output = fe_output.squeeze().permute(1, 2, 0)
                    fe_output = fe_output.detach().cpu().data.numpy()

                    # Extract LOI results
                    if args.loi_level == 'pixel':
                        loi_results = loi_model.pixelwise_forward(cc_output, fe_output)
                    elif args.loi_level == 'region':
                        loi_results = loi_model.regionwise_forward(cc_output, fe_output)
                    elif args.loi_level == 'crossed':
                        loi_results = loi_model.cross_pixelwise_forward(cc_output, fe_output)
                    else:
                        print('Incorrect LOI level')
                        exit()

                    # Get 2 frame results and sum
                    # @TODO: Here is switch between sides. Correct this!!!!!!
                    total_d1 += sum(loi_results[1])
                    total_d2 += sum(loi_results[0])


                    ucsd_total_count[0].append(sum(loi_results[1]))
                    ucsd_total_count[1].append(sum(loi_results[0]))

                    totals = [total_d1, total_d2]

                    # Save every frame
                    per_frame[0].append(sum(loi_results[1]))
                    per_frame[1].append(sum(loi_results[0]))

                    # Update GUI
                    pbar.set_description('{} ({}), {} ({})'.format(totals[0], crosses[0], totals[1], crosses[1]))
                    pbar.update(1)


                    if v_i == 0 and l_i == 0:

                        # Save for debugging
                        if s_i == 0:
                            cc_img = Image.fromarray(cc_output * 255.0 / cc_output.max())
                            cc_img = cc_img.convert("L")
                            cc_img.save('counting.png')

                            frame1_img = frame_pair.get_frames(0).get_image()
                            total_image = frame1_img.copy().convert('RGB')
                            total_image.save('orig.png')


                        if s_i < 10:
                            img = Image.open(video.get_frame_pairs()[s_i].get_frames(0).get_image_path())

                            density = torch.FloatTensor(density_filter.gaussian_filter_fixed_density(video.get_frame_pairs()[s_i].get_frames(0), 8))
                            density = density.numpy()
                            utils.save_loi_sample("{}_{}_{}".format(v_i, l_i, s_i), img, density, fe_output)

                    metrics['timing'].update(timer.show(False))


                pbar.close()

                # Fixes error in UCSD mix
                ucsd_total_count[0].append(0.0)
                ucsd_total_count[1].append(0.0)

                print("Timing {}".format(metrics['timing'].avg))

                t_left, t_right = crosses
                p_left, p_right = totals
                mae = abs(t_left - p_left) + abs(t_right - p_right)
                metrics['loi_mae'].update(mae)
                mse = math.pow(t_left - p_left, 2) + math.pow(t_right - p_right, 2)
                metrics['loi_mse'].update(mse)
                total_mae = abs(t_left + t_right - (p_left + p_right))
                total_mse = math.pow(t_left + t_right - (p_left + p_right), 2)
                percentual_total_mae = (p_left + p_right) / (t_left + t_right)
                relative_mae = mae / (t_left + t_right)
                metrics['loi_ptmae'].update(percentual_total_mae)
                metrics['loi_rmae'].update(relative_mae)
                print("LOI performance (MAE: {}, MSE: {}, TMAE: {}, TMSE: {}, PTMAE: {})".format(mae, mse, total_mae,
                                                                                                 total_mse,
                                                                                                 percentual_total_mae))

                results.append({
                    'vid': v_i,
                    'loi': l_i,
                    'mae': mae,
                    'mse': mse,
                    'ptmae': percentual_total_mae,
                    'rmae': relative_mae
                })

                if args.dataset == 'dam':

                    results = {'per_frame': per_frame}

                    with open('dam_results_{}_{}.json'.format(v_i, l_i), 'w') as outfile:
                        json.dump(results, outfile)

            #break

        if args.dataset == 'ucsd':
            ucsd_total_gt = ucsdpeds.load_countings('../data/ucsdpeds')
            ucsd_total_count2 = ucsd_total_count
            ucsd_total_count = [[], []]

            for i in range(len(ucsd_total_count2[0])):
                for _ in range(args.frames_between):
                    ucsd_total_count[0].append(ucsd_total_count2[0][i] / args.frames_between)
                    ucsd_total_count[1].append(ucsd_total_count2[1][i] / args.frames_between)

            wmae = [[], []]
            tmae = [[], []]
            imae = [[], []]

            for i, _ in enumerate(ucsd_total_count[0]):
                imae[0].append(abs(ucsd_total_count[0][i] - ucsd_total_gt[0][i]))
                imae[1].append(abs(ucsd_total_count[1][i] - ucsd_total_gt[1][i]))

                if i >= 600:
                    tmae[0].append(abs(sum(ucsd_total_count[0][600:i + 1]) - sum(ucsd_total_gt[0][600:i + 1])))
                    tmae[1].append(abs(sum(ucsd_total_count[1][600:i + 1]) - sum(ucsd_total_gt[1][600:i + 1])))

                    if i + 100 < 1200:
                        wmae[0].append(abs(sum(ucsd_total_count[0][i:i + 100]) - sum(ucsd_total_gt[0][i:i + 100])))
                        wmae[1].append(abs(sum(ucsd_total_count[1][i:i + 100]) - sum(ucsd_total_gt[1][i:i + 100])))
                else:
                    tmae[0].append(abs(sum(ucsd_total_count[0][:i + 1]) - sum(ucsd_total_gt[0][:i + 1])))
                    tmae[1].append(abs(sum(ucsd_total_count[1][:i + 1]) - sum(ucsd_total_gt[1][:i + 1])))

                    if i+100 < 600:
                        wmae[0].append(abs(sum(ucsd_total_count[0][i:i + 100]) - sum(ucsd_total_gt[0][i:i + 100])))
                        wmae[1].append(abs(sum(ucsd_total_count[1][i:i + 100]) - sum(ucsd_total_gt[1][i:i + 100])))



            print("UCSD results, total error left: {}, right: {}".format(abs(sum(ucsd_total_count[0]) - sum(ucsd_total_gt[0])), abs(sum(ucsd_total_count[1]) - sum(ucsd_total_gt[1]))))
            print("IMAE: {} | {}".format(sum(imae[0]) / len(imae[0]), sum(imae[1]) / len(imae[1])))
            print("TMAE: {} | {}".format(sum(tmae[0]) / len(tmae[0]), sum(tmae[1]) / len(tmae[1])))
            print("WMAE: {} | {}".format(sum(wmae[0]) / len(wmae[0]), sum(wmae[1]) / len(wmae[1])))

        print("MAE: {}, MSE: {}, PTMAE: {}".format(metrics['loi_mae'].avg,
                                                   metrics['loi_mse'].avg,
                                                   metrics['loi_ptmae'].avg))

        results = {'loi_mae': metrics['loi_mae'].avg, 'loi_mse': metrics['loi_mse'].avg, 'loi_ptmae': metrics['loi_ptmae'].avg,\
               'roi_mae': metrics['mae'].avg, 'roi_mse': metrics['mse'].avg, 'loi_rmae': metrics['loi_rmae'].avg}  # ROI (First Line is LOI)

        outname = 'all_{}_{}_{}_{}_{}'.format(args.dataset, args.model, args.loi_level, args.loi_maxing, datetime.now().strftime("%Y%m%d_%H%M%S"))
        with open('loi_results/{}.json'.format(outname), 'w') as outfile:
            json.dump(results, outfile)

        return results


if __name__ == '__main__':
    args = parser.parse_args()
    args.seed = time.time()
    args.batch_size = 1
    args.dataloader_workers = 2  # 0 for reproducability

    args.loss_function = 'L2'  # L2
    args.optimizer = 'adam'

    args.resize_to_orig = True  # Resize output to orig. (Or groundtruth to output)

    args.print_every = 40  # Print every x amount of minibatches
    args.test_epochs = 1  # Run every tenth epoch a test

    args.train_split = 4
    args.cross_val_amount = 50
    args.train_amount = 200

    args.single_dataset = False

    # Add date and time so we can just run everything very often :)
    args.save_dir = args.name # '{}_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"), args.name)

    args.loi_flow_smoothing = False
    args.loi_flow_width = False

    args.real_loi_version = args.loi_version

    if args.loi_version == 'v3':
        args.loi_version = 'v2'
        args.loi_flow_width = True
    elif args.loi_version == 'v4.6':
        args.loi_version = 'v2'
        args.loi_flow_width = True
        args.loi_flow_smoothing = True


    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    if args.pre != '':
        args.pre = 'weights/{}/best_model.pt'.format(args.pre)

    if args.mode == 'loi':
        print(loi_test(args))
    else:
        train(args)
        # args.pre = ''
        # loi_test(args)
