import argparse
import math
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

from model import DRNetModel
from dataset import DRNetDataset

from pathlib import Path
import utils

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from datasets import shanghaitech, fudan

parser = argparse.ArgumentParser(description='PyTorch DDFlow (PWCNet backbone)')

parser.add_argument('name', metavar='NAME', type=str,
                    help='Used as postfix for the save directory')

def train(args):
    print('Initializing dataset...')

    train_frames = shanghaitech.load_all_frames('../data/ShanghaiTech/part_A_final/train_data')
    print("Loaded {} trainings frames".format(len(train_frames)))
    test_frames = shanghaitech.load_all_frames('../data/ShanghaiTech/part_A_final/test_data')
    print("Loaded {} testing frames".format(len(test_frames)))

    writer = SummaryWriter(log_dir='summaries/{}'.format(args.save_dir))
    Path('results/{}/'.format(args.save_dir)).mkdir(parents=True, exist_ok=True)
    Path('train_results/{}/'.format(args.save_dir)).mkdir(parents=True, exist_ok=True)

    train_dataset = DRNetDataset(train_frames, args.density_model, args.patch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    print('Initializing model...')
    model = DRNetModel().cuda()
    criterion = nn.L1Loss(reduction='mean').cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)

    o = 0
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
            pred_densities = model(images)
            #print(pred_densities.shape)

            loss = criterion(pred_densities, densities)
            loss.backward()
            optimizer.step()

            pred_densities = pred_densities.detach()
            if i == 0 and epoch%args.test_epochs == args.test_epochs-1:
                save_image(images[0], 'train_results/{}img_{}.png'.format(args.save_dir, epoch))
                save_image(utils.norm_to_img(densities[0]), 'train_results/{}/true_{}.png'.format(args.save_dir, epoch))
                save_image(utils.norm_to_img(pred_densities[0]), 'train_results/{}/pred_{}.png'.format(args.save_dir, epoch))


            running_loss += loss.item()

            o += 1
            writer.add_scalar('CC/Loss/train', loss.item(), o)

            # print every 2000 mini-batches
            if i % args.print_every == args.print_every-1:
                print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / args.print_every))
                running_loss = 0.0

        if epoch%args.test_epochs == args.test_epochs-1:
            avg, avg_sq = test_run(epoch, test_frames, model, args.density_model)
            writer.add_scalar('CC/MAE/train', avg, epoch)
            writer.add_scalar('CC/MSE/train', avg_sq, epoch)

    return

def test_run(epoch, frames, model, density_model):
    print("Do testrun {}:".format(epoch))
    test_dataset = DRNetDataset(frames, density_model)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    avg = utils.AverageMeter()
    avg_sq = utils.AverageMeter()

    truth = utils.AverageMeter()
    pred = utils.AverageMeter()

    model.eval()

    for i, batch in enumerate(test_loader):
        images, densities = batch
        images = images.cuda()
        densities = densities.cuda()

        orig_shape = images.shape

        int_preprocessed_width = int(math.floor(math.ceil(images.shape[3] / args.resize_diff) * args.resize_diff))
        int_preprocessed_height = int(math.floor(math.ceil(images.shape[2] / args.resize_diff) * args.resize_diff))

        # Resize to get a size which fits into the network
        images = F.interpolate(input=images,
                             size=(int_preprocessed_height, int_preprocessed_width),
                             mode='bilinear', align_corners=False)

        predictions = model(images)
        predictions = predictions.detach()

        predictions = F.interpolate(input=predictions,
                                  size=(orig_shape[2], orig_shape[3]),
                                  mode='bilinear', align_corners=False)

        predictions /= 100.
        densities /= 100.

        truth.update(densities.sum().item())
        pred.update(predictions.sum().item())

        avg.update(abs((predictions.sum() - densities.sum()).item()))
        avg_sq.update(torch.pow(predictions.sum() - densities.sum(), 2).item())

        if i == 0:
            save_image(images[0], 'results/{}/img_{}.png'.format(args.save_dir, epoch))
            save_image(utils.norm_to_img(densities[0]), 'results/{}/true_{}.png'.format(args.save_dir, epoch))
            save_image(utils.norm_to_img(predictions[0]), 'results/{}/pred_{}.png'.format(args.save_dir, epoch))

    print("--- RESULTS [MAE: {}, RMSE: {}]".format(avg.avg, math.pow(avg_sq.avg, 0.5)))
    #print("Avg truth: {}| avg pred: {}".format(truth.avg, pred.avg))
    model.train()

    return avg.avg, avg_sq.avg


if __name__ == '__main__':
    args = parser.parse_args()
    args.epochs = 1000
    args.print_every = 50  # Print every x amount of minibatches
    args.patch_size = (128, 128)

    args.save_dir = '{}_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"), args.name)
    #args.patch_size = (256, 256)
    args.density_model = 'fixed-8'

    args.resize_diff = 64.0

    args.test_epochs = 1  # Run every fifth epoch a test
    train(args)
