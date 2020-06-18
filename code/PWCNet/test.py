#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys

from matplotlib import pyplot as plt
import cv2

import model
import utils

from dataset import SimpleDataset


# Add base path to import dir for importing datasets
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from datasets import tub

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'default'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use

def estimate(net, tenFirst, tenSecond):
	intWidth = tenFirst.shape[3]
	intHeight = tenFirst.shape[2]

	tenPreprocessedFirst = tenFirst.cuda()
	tenPreprocessedSecond = tenSecond.cuda()

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

	tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	result = net(tenPreprocessedFirst, tenPreprocessedSecond)

	multiplication = (intHeight/result.shape[2]) * (intWidth/result.shape[3])

	tenFlow = multiplication * torch.nn.functional.interpolate(input=result, size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	return tenFlow.cpu()

if __name__ == '__main__':
	netNetwork = model.PWCNet()
	netNetwork.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load('network-' + arguments_strModel + '.pytorch').items()})
	netNetwork = netNetwork.cuda().eval()

	videos = tub.load_all_videos('../data/TUBCrowdFlow')
	frame_pairs = videos[0].get_frame_pairs()
	frame_pairs = frame_pairs[0:50]

	dataset = SimpleDataset(frame_pairs)
	batch_size = 2
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=2)

	timer = utils.sTimer('total loading')
	for i, (tenFirst, tenSecond) in enumerate(dataloader):

		tenOutput = estimate(netNetwork, tenFirst, tenSecond)
		flow_np = tenOutput.detach().numpy().transpose(0, 2, 3, 1)

		print(i)

		for o in range(flow_np.shape[0]):
			num = batch_size * i + o
			# rgb = utils.flo_to_color(flow_np[o])
			# plt.imsave('results/flow_{}.png'.format(num), rgb)

		objOutput = open('results/flow_{}.flo'.format(i), 'wb')
		numpy.array([80, 73, 69, 72], numpy.uint8).tofile(objOutput)
		numpy.array([flow_np.shape[0], flow_np.shape[2], flow_np.shape[1]], numpy.int32).tofile(objOutput)
		numpy.array(flow_np, numpy.float32).tofile(objOutput)
		objOutput.close()

	timer.show()
# end