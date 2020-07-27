import argparse
import time
import math
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

from model import DRNetModel
from dataset import BasicDataset

from pathlib import Path
import utils

# Add base path to import dir for importing datasets
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from datasets import tub, shanghaitech

parser = argparse.ArgumentParser(description='CrowdCounting (PWCNet backbone)')

parser.add_argument('name', metavar='NAME', type=str,
                    help='Used as postfix for the save directory')

parser.add_argument('--mode', '-m', metavar='MODE', type=str, default='train',
                    help='Train or test')

parser.add_argument('--pre', '-p', metavar='PRETRAINED_MODEL', default='weights/20200722_091354_v02_csr_sgd_improved/best_model.pt', type=str,
                    help='Path to the TUB dataset')

parser.add_argument('--data_path', '-d', metavar='DATA_PATH', default='../data/TUBCrowdFlow', type=str,
                    help='Path to the TUB dataset')


def save_sample(args, dir, info, density, true, img):
    save_image(img, '{}/{}/img.png'.format(dir, args.save_dir))
    save_image(true / true.max(), '{}/{}/true.png'.format(dir, args.save_dir))
    save_image(density / density.max(), '{}/{}/pred_{}.png'.format(dir, args.save_dir, info))


def load_dataset(args):
    # Get dataset and dataloader
    # train_frames = []
    # val_frames = []
    # for video in tub.load_all_videos(args.data_path, load_peds=False):
    #     train_video, _, val_video, _, _, _ = tub.train_val_test_split(video, None)
    #     train_frames += train_video.get_frames()
    #     val_frames += val_video.get_frames()
    #
    # val_frames = val_frames[::10]

    train_frames = shanghaitech.load_all_frames('../data/ShanghaiTech/part_A_final/train_data', load_labeling=False)
    val_frames = shanghaitech.load_all_frames('../data/ShanghaiTech/part_A_final/test_data', load_labeling=False)

    print("Loaded {} trainings frames".format(len(train_frames)))
    print("Loaded {} testing frames".format(len(val_frames)))

    return (BasicDataset(train_frames, args.density_model, args.resize_diff, True),
            BasicDataset(val_frames, args.density_model, args.resize_diff, False))


import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from CSRNet.model import CSRNet
from DDFlow_pytorch.model import Extractor

from models import ModelV1


def load_model(args):
    model = CSRNet(load_weights=True).cuda()
    extractor = Extractor().cuda()

    if args.pre:
        model.load_state_dict(torch.load(args.pre))

    return model, extractor

def run_model(input, model, extractor):
    # features = extractor(input)
    out = model(input)
    return out


def train(args):
    print('Initializing result storage...')
    writer = SummaryWriter(log_dir='summaries/{}'.format(args.save_dir))
    Path('weights/{}/'.format(args.save_dir)).mkdir(parents=True, exist_ok=True)
    Path('results/{}/'.format(args.save_dir)).mkdir(parents=True, exist_ok=True)
    #Path('train_results/{}/'.format(args.save_dir)).mkdir(parents=True, exist_ok=True)

    print('Initializing dataset...')
    train_dataset, test_dataset = load_dataset(args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.dataloader_workers)

    print('Initializing model...')
    model, extractor = load_model(args)
    criterion = nn.MSELoss(reduction='sum').cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    # optimizer = optim.SGD(model.parameters(), 1e-6,
    #                             momentum=0.95,
    #                             weight_decay=5*1e-4)

    best_mae = None
    print('Start training...')
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            images, densities = batch
            images = images.cuda()
            densities = densities.cuda()

            # Set grad to zero
            optimizer.zero_grad()

            # Run model and optimize
            pred_densities = run_model(images, model, extractor)
            factor = (densities.shape[2]*densities.shape[3]) / (pred_densities.shape[2]*pred_densities.shape[3])

            densities = F.interpolate(input=densities,
                                   size=(pred_densities.shape[2], pred_densities.shape[3]),
                                   mode='bicubic', align_corners=False) * factor

            loss = criterion(pred_densities, densities)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # print every 2000 mini-batches
            if i % args.print_every == args.print_every - 1:
                print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / args.print_every))

        writer.add_scalar('Train/Loss', running_loss, epoch)
        if epoch % args.test_epochs == args.test_epochs - 1:
            timer = utils.sTimer('Test run')
            avg, avg_sq = test_run(args, epoch, test_dataset, model)
            writer.add_scalar('Val/eval_time', timer.show(False), epoch)
            writer.add_scalar('Val/MAE', avg.avg, epoch)
            writer.add_scalar('Val/MSE', avg_sq.avg, epoch)

            torch.save(extractor.state_dict(), 'weights/{}/last_extractor.pt'.format(args.save_dir))
            torch.save(model.state_dict(), 'weights/{}/last_model.pt'.format(args.save_dir))
            if best_mae is None or best_mae > avg.avg:
                best_mae = avg.avg
                torch.save(extractor.state_dict(), 'weights/{}/best_extractor.pt'.format(args.save_dir))
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
        images, densities = batch
        images = images.cuda()
        densities = densities.cuda()

        predictions = model(images).detach()

        truth.update(densities.sum().item())
        pred.update(predictions.sum().item())

        avg.update(abs((predictions.sum() - densities.sum()).item()))
        avg_sq.update(torch.pow(predictions.sum() - densities.sum(), 2).item())

        if i == 1 and save:
            predictions = F.interpolate(input=predictions,
                                        size=(images.shape[2], images.shape[3]),
                                        mode='bilinear', align_corners=False)

            save_sample(args, 'results', epoch, predictions[0], densities[0], images[0])

    print("--- TEST [MAE: {}, RMSE: {}]".format(avg.avg, math.pow(avg_sq.avg, 0.5)))
    model.train()

    return avg, avg_sq


if __name__ == '__main__':
    args = parser.parse_args()
    args.batch_size = 1

    # Keep these fixed to make sure reproducibility
    args.dataloader_workers = 1
    args.seed = time.time()

    args.epochs = 2000
    args.print_every = 300  # Print every x amount of minibatches
    args.patch_size = (128, 128)

    # Add date and time so we can just run everything very often :)
    args.save_dir = '{}_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"), args.name)

    # args.patch_size = (256, 256)
    args.density_model = 'flex'  # 'fixed-8'

    args.resize_diff = 1.0

    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    args.test_epochs = 4  # Run every fifth epoch a test
    if args.mode == 'test':
        test(args)
    else:
        train(args)
