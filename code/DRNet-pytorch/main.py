import argparse

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import DRNetModel
from dataset import DRNetDataset

def train(args):
    print('Initializing dataset...')
    dataset = DRNetDataset()
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    print('Initializing model...')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    model = DRNetModel()

    # @TODO Do something with pytorch visualization

    print('Start training...')
    for epoch in range(args.epochs):

        running_loss = 0.0
        for i, batch in enumerate(trainloader, 0):
            images, densities = batch

            # Set grad to zero
            optimizer.zero_grad()

            # Run model and optimize
            pred_densities = model(images)
            loss = criterion(pred_densities, densities)
            loss.backward()
            optimizer.step()


            running_loss += loss.item()
            # print every 2000 mini-batches
            if i % args.print_every == args.print_every-1:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / args.print_every))
                running_loss = 0.0

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 2
    args.print_every = 2000 # Print every x amount of minibatches
    train(args)