import argparse
import time
import math
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

# from model import DRNetModel
from dataset import SimpleDataset
from models import ModelV2, ModelV3, ModelV31

from pathlib import Path
import utils

# Add base path to import dir for importing datasets
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from datasets import tub, shanghaitech
from DDFlow_pytorch import losses
# from CSRNet.model import CSRNet

parser = argparse.ArgumentParser(description='CrowdCounting (PWCNet backbone)')

parser.add_argument('name', metavar='NAME', type=str,
                    help='Used as postfix for the save directory')

parser.add_argument('--mode', '-m', metavar='MODE', type=str, default='train',
                    help='Train or test')

parser.add_argument('--pre', '-p', metavar='PRETRAINED_MODEL', default='', type=str,
                    help='Path to the TUB dataset')

parser.add_argument('--data_path', '-d', metavar='DATA_PATH', default='../data/TUBCrowdFlow', type=str,
                    help='Path to the TUB dataset')


def save_sample(args, dir, info, density, true, img, flow):
    save_image(img, '{}/{}/img.png'.format(dir, args.save_dir))
    save_image(true / true.max(), '{}/{}/true.png'.format(dir, args.save_dir))
    save_image(density / density.max(), '{}/{}/pred_{}.png'.format(dir, args.save_dir, info))
    plt.imsave('{}/{}/flow_{}.png'.format(dir, args.save_dir, info), flow)


def load_dataset(args, between=1):
    # Get dataset and dataloader
    train_pairs = []
    val_pairs = []
    for video in tub.load_all_videos(args.data_path, load_peds=False):
        train_video, _, val_video, _, _, _ = tub.train_val_test_split(video, None)
        train_video.generate_frame_pairs(between)
        train_pairs += train_video.get_frame_pairs()

        val_video.generate_frame_pairs(between)
        val_pairs += val_video.get_frame_pairs()

    val_pairs = val_pairs[::10]

    print("Loaded {} trainings frames".format(len(train_pairs)))
    print("Loaded {} testing frames".format(len(val_pairs)))

    return (SimpleDataset(train_pairs, args.density_model, True),
            SimpleDataset(val_pairs, args.density_model, False))

def load_model(args):
    model = ModelV31(load_pretrained=True).cuda()

    if args.pre:
        model.load_state_dict(torch.load(args.pre))

    return model

def train(args):
    print('Initializing result storage...')
    writer = SummaryWriter(log_dir='summaries/{}'.format(args.save_dir))
    Path('weights/{}/'.format(args.save_dir)).mkdir(parents=True, exist_ok=True)
    Path('results/{}/'.format(args.save_dir)).mkdir(parents=True, exist_ok=True)

    print('Initializing dataset...')
    train_dataset, test_dataset = load_dataset(args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.dataloader_workers)

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
        optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0)
    else:
        optimizer = optim.SGD(model.parameters(), 1e-6, momentum=0.95, weight_decay=5*1e-4)

    best_mae = None
    print('Start training...')
    for epoch in range(args.epochs):
        running_cc_loss = utils.AverageMeter()
        running_fe_loss = utils.AverageMeter()
        running_total_loss = utils.AverageMeter()
        for i, batch in enumerate(train_loader):

            frames1, frames2, densities = batch
            frames1 = frames1.cuda()
            frames2 = frames2.cuda()
            densities = densities.cuda()

            # TODO Some resizing for correctly inputting in the model

            # Set grad to zero
            optimizer.zero_grad()

            # Run model and optimize
            flow_fw, flow_bw, pred_densities = model(frames1, frames2)

            # @TODO Resizing back to original sizes
            factor = (densities.shape[2]*densities.shape[3]) / (pred_densities.shape[2]*pred_densities.shape[3])

            if epoch == 0 and i == 0:
                print("Resize factor: {}".format(factor))

            densities = F.interpolate(input=densities,
                                   size=(pred_densities.shape[2], pred_densities.shape[3]),
                                   mode='bicubic', align_corners=False) * factor

            if args.loss_focus != 'cc':
                photo_losses = losses.create_photometric_losses(frames1, frames2, flow_fw, flow_bw)
                fe_loss = photo_losses['abs_robust_mean']['no_occlusion']

            if args.loss_focus != 'fe':
                cc_loss = criterion(pred_densities, densities)

            if args.loss_focus == 'cc':
                loss = cc_loss
            elif args.loss_focus == 'fe':
                loss = fe_loss
            else:
                loss = fe_loss + cc_loss * args.cc_weight

            loss.backward()
            optimizer.step()

            running_total_loss.update(loss.item())
            running_cc_loss.update(cc_loss.item())
            running_fe_loss.update(fe_loss.item())

            # print every 2000 mini-batches
            if i % args.print_every == args.print_every - 1:
                print('[%d, %5d] loss: %.5f | %.5f | %.5f' % (epoch + 1, i + 1,
                                                              running_total_loss.avg, running_fe_loss.avg,
                                                              running_cc_loss.avg))

        if epoch > 0:
            writer.add_scalar('Train/Total_loss', running_total_loss.avg, epoch)
            writer.add_scalar('Train/FE_loss', running_fe_loss.avg, epoch)
            writer.add_scalar('Train/CC_loss', running_cc_loss.avg, epoch)

        if epoch % args.test_epochs == args.test_epochs - 1:
            timer = utils.sTimer('Test run')
            avg, avg_sq = test_run(args, epoch, test_dataset, model)
            writer.add_scalar('Val/eval_time', timer.show(False), epoch)
            writer.add_scalar('Val/MAE', avg.avg, epoch)
            writer.add_scalar('Val/MSE', avg_sq.avg, epoch)
            torch.save(model.state_dict(), 'weights/{}/last_model.pt'.format(args.save_dir))
            if best_mae is None or best_mae > avg.avg:
                best_mae = avg.avg
                torch.save(model.state_dict(), 'weights/{}/best_model.pt'.format(args.save_dir))
                print("----- NEW BEST!! -----")
    return

def test(args):
    _, dataset = load_dataset(args)
    model = load_model(args)
    avg, avg_sq = test_run(args, 1, dataset, model, save=False)


def test_run(args, epoch, test_dataset, model, save=True):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=args.dataloader_workers)

    avg = utils.AverageMeter()
    avg_sq = utils.AverageMeter()

    truth = utils.AverageMeter()
    pred = utils.AverageMeter()

    model.eval()

    for i, batch in enumerate(test_loader):
        frames1, frames2, densities = batch
        frames1 = frames1.cuda()
        frames2 = frames2.cuda()
        densities = densities.cuda()

        flow_fw, flow_bw, pred_densities = model(frames1, frames2)
        flow_fw = flow_fw.detach()
        flow_bw = flow_bw.detach()
        pred_densities = pred_densities.detach()

        truth.update(densities.sum().item())
        pred.update(pred_densities.sum().item())

        avg.update(abs((pred_densities.sum() - densities.sum()).item()))
        avg_sq.update(torch.pow(pred_densities.sum() - densities.sum(), 2).item())

        if i == 1 and save:
            pred_densities = F.interpolate(input=pred_densities,
                                        size=(frames1.shape[2], frames1.shape[3]),
                                        mode='bicubic', align_corners=False)

            flow_fw = flow_fw.detach().cpu().numpy().transpose(0, 2, 3, 1)
            rgb = utils.flo_to_color(flow_fw[0])

            save_sample(args, 'results', epoch, pred_densities[0], densities[0], frames1[0], rgb)

    print("--- TEST [MAE: {}, RMSE: {}]".format(avg.avg, math.pow(avg_sq.avg, 0.5)))
    model.train()

    return avg, avg_sq


if __name__ == '__main__':
    args = parser.parse_args()
    args.seed = time.time()
    args.batch_size = 1
    args.dataloader_workers = 1 # 0 for reproducability

    args.loss_function = 'L1'  # L2
    args.optimizer = 'adam'

    args.loss_focus = 'both' # 'cc' or 'fe'
    args.cc_weight = 0.2  # equal is around 0.002

    args.epochs = 500
    args.print_every = 50  # Print every x amount of minibatches
    #args.patch_size = (128, 128)

    # Add date and time so we can just run everything very often :)
    args.save_dir = '{}_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"), args.name)

    # args.patch_size = (256, 256)
    args.density_model = 'fixed-5'  # 'fixed-8' / 'flex'

    args.resize_diff = 1.0

    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    args.test_epochs = 5  # Run every tenth epoch a test
    if args.mode == 'test':
        test(args)
    else:
        train(args)
