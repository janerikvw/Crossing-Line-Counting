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
from models import V3Adapt, V3Correlation, V3EndFlow
from PIL import Image, ImageDraw
from tqdm import tqdm

from pathlib import Path
import utils

# Add base path to import dir for importing datasets
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from datasets import basic_entities, fudan
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

parser.add_argument('--model', metavar='MODEL', default='v3correlation', type=str,
                    help='Which model gonna train')

parser.add_argument('--resize_patch', metavar='RESIZE_PATCH', default='on', type=str,
                    help='Resizing patch so zoom/not zoom')


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
    for video_path in glob('../data/Fudan/train_data/*/'):
        video = fudan.load_video(video_path)
        total_num += len(video.get_frames())
        splitted_frames = n_split_pairs(video.get_frames(), args.train_split, args.frames_between, skip_inbetween=False)

        # Possibly overfit slightly, because middle parts could almost overlap with outer parts, so shuffle to balance
        random.shuffle(splitted_frames)
        for i, split in enumerate(splitted_frames):
            splits[i] += split

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

    train_pairs = random.sample(train_pairs, args.train_amount)
    cross_pairs = random.sample(cross_pairs, args.cross_val_amount)

    return (SimpleDataset(train_pairs, args, True),
            SimpleDataset(cross_pairs, args, False))


def load_test_dataset(args):
    test_vids = []
    for video_path in glob('../data/Fudan/test_data/*/'):
        test_vids.append(fudan.load_video(video_path))
    return test_vids


def load_model(args):
    model = None
    if args.model == 'v3adaptive':
        model = V3Adapt(load_pretrained=True).cuda()
    elif args.model == 'csrnet':
        model = CSRNet().cuda()
        args.loss_focus = 'cc'
    elif args.model == 'v3correlation':
        model = V3Correlation(load_pretrained=True).cuda()
    elif args.model == 'v3endflow':
        model = V3EndFlow(load_pretrained=True).cuda()
    else:
        print("Error! Incorrect model selected")
        exit()

    if args.pre:
        print("Load pretrained model:", args.pre)
        model.load_state_dict(torch.load(args.pre))

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
        # CSR: optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
        #optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0)
    else:
        optimizer = optim.SGD(model.parameters(), 1e-6, momentum=0.95, weight_decay=5*1e-4)


    best_mae = None
    best_mse = None
    print('Start training...')

    if args.single_dataset:
        train_dataset, test_dataset = setup_train_cross_dataset(train_pair_splits, 0, args)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.dataloader_workers)

    for epoch in range(args.epochs):

        if not args.single_dataset:
            train_dataset, test_dataset = setup_train_cross_dataset(train_pair_splits, epoch, args)
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

            if args.resize_to_orig:
                pred_densities = F.interpolate(input=pred_densities,
                                               size=(densities.shape[2], densities.shape[3]),
                                               mode='bicubic', align_corners=False) / factor
            else:
                densities = F.interpolate(input=densities,
                                               size=(pred_densities.shape[2], pred_densities.shape[3]),
                                               mode='bicubic', align_corners=False) * factor

            if args.loss_focus != 'cc':
                photo_losses = losses.create_photometric_losses(frames1, frames2, flow_fw, flow_bw)
                fe_loss = photo_losses['abs_robust_mean']['no_occlusion']

                loss_container['abs_no_occlusion'].update(photo_losses['abs_robust_mean']['no_occlusion'].item())
                loss_container['abs_occlusion'].update(photo_losses['abs_robust_mean']['occlusion'].item())
                loss_container['census_no_occlusion'].update(photo_losses['census']['no_occlusion'].item())
                loss_container['census_occlusion'].update(photo_losses['census']['occlusion'].item())

            if args.loss_focus != 'fe':
                cc_loss = criterion(pred_densities, densities)

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
            avg, avg_sq = test_run(args, epoch, test_dataset, model)
            writer.add_scalar('Val/eval_time', timer.show(False), epoch)
            writer.add_scalar('Val/MAE', avg.avg, epoch)
            writer.add_scalar('Val/MSE', avg_sq.avg, epoch)
            torch.save(model.state_dict(), 'weights/{}/last_model.pt'.format(args.save_dir))
            if best_mae is None or best_mae > avg.avg:
                best_mae = avg.avg
                best_mse = avg_sq.avg
                torch.save(model.state_dict(), 'weights/{}/best_model.pt'.format(args.save_dir))
                print("----- NEW BEST!! -----")

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

    truth = utils.AverageMeter()
    pred = utils.AverageMeter()

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

            truth.update(densities.sum().item())
            pred.update(pred_densities.sum().item())

            avg.update(abs((pred_densities.sum() - densities.sum()).item()))
            avg_sq.update(torch.pow(pred_densities.sum() - densities.sum(), 2).item())

            if i == 1 and save:
                pred_densities = F.interpolate(input=pred_densities,
                                            size=(frames1.shape[2], frames1.shape[3]),
                                            mode='bicubic', align_corners=False)
                if args.loss_focus != 'cc':
                    flow_fw = flow_fw.detach().cpu().numpy().transpose(0, 2, 3, 1)
                    rgb = utils.flo_to_color(flow_fw[0])
                else:
                    rgb = None

                save_sample(args, 'results', epoch, pred_densities[0], densities[0], frames1[0], rgb)

    print("--- TEST [MAE: {}, RMSE: {}]".format(avg.avg, math.pow(avg_sq.avg, 0.5)))
    model.train()

    return avg, avg_sq

import loi
def loi_test(args):
    metrics = utils.AverageContainer()

    # args.save_dir = '20200907_065813_dataset-fudan_epochs-250_resize_patch-on'
    # args.model = 'v3correlation

    args.save_dir = '20200909_184358_dataset-fudan_model-csrnet_epochs-250_loss_focus-cc'
    args.model = 'csrnet'
    args.loss_focus = 'cc'

    if args.pre == '':
        args.pre = 'weights/{}/best_model.pt'.format(args.save_dir)
    model = load_model(args)
    model.eval()
    if args.loss_focus == 'cc':
        fe_model = V3Correlation(load_pretrained=True).cuda()
        fe_model.load_state_dict(
            torch.load('weights/20200907_065813_dataset-fudan_epochs-250_resize_patch-on/last_model.pt')
        )
        fe_model.eval()

    with torch.no_grad():
        test_vids = load_test_dataset(args)
        for v_i, video in enumerate(test_vids):

            print("Video:", video.get_path())

            video.generate_frame_pairs(distance=args.frames_between, skip_inbetween=True)
            dataset = SimpleDataset(video.get_frame_pairs(), args, False)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.dataloader_workers)

            for l_i, line in enumerate(video.get_lines()):
                image = video.get_frame(0).get_image()
                width, height = image.size
                image.close()
                point1, point2 = line.get_line()

                loi_model = loi.LOI_Calculator(point1, point2,
                                               img_width=width, img_height=height,
                                               crop_processing=False)
                loi_model.create_regions()

                total_d1 = 0
                total_d2 = 0

                pbar = tqdm(total=len(video.get_frame_pairs()))
                for s_i, batch in enumerate(dataloader):
                    torch.cuda.empty_cache()

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

                    # Resize and save as numpy
                    cc_output = loi_model.to_orig_size(cc_output)
                    cc_output = cc_output.squeeze().squeeze()
                    cc_output = cc_output.detach().cpu().data.numpy()
                    fe_output = loi_model.to_orig_size(fe_output)
                    fe_output = fe_output.squeeze().permute(1, 2, 0)
                    fe_output = fe_output.detach().cpu().data.numpy()

                    # Extract LOI results
                    loi_results = loi_model.regionwise_forward(cc_output, fe_output)

                    # Get 2 frame results and sum
                    total_d1 += sum(loi_results[0])
                    total_d2 += sum(loi_results[1])
                    totals = [total_d1, total_d2]

                    # Update GUI
                    crosses = line.get_crossed()
                    pbar.set_description('{} ({}), {} ({})'.format(totals[0], crosses[1], totals[1], crosses[0]))
                    pbar.update(1)

                    # Save for debugging
                    # cc_img = Image.fromarray(cc_output * 255.0 / cc_output.max())
                    # cc_img = cc_img.convert("L")
                    # cc_img.save('counting.png')
                    #
                    # frame1_img = frame_pair.get_frames(0).get_image()
                    # total_image = frame1_img.copy().convert('RGB')
                    # total_image.save('orig.png')

                pbar.close()
                print("ROI performance (MAE:", metrics['mae'].avg, "MSE:", metrics['mse'].avg,")")
            if v_i == 2:
                break


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

    args.train_split = 3
    args.cross_val_amount = 50
    args.train_amount = 200

    args.single_dataset = False

    # Add date and time so we can just run everything very often :)
    args.save_dir = args.name # '{}_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"), args.name)


    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    if args.mode == 'loi':
        loi_test(args)
    else:
        train(args)
        args.pre = ''
        loi_test(args)
