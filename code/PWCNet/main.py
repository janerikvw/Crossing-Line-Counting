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


# Add base path to import dir for importing datasets
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from datasets import tub

try:
	from .correlation import correlation # the custom cost volume layer
except:
	sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python
# end

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'default'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use

def estimate(net, tenFirst, tenSecond):
	intWidth = tenFirst.shape[2]
	intHeight = tenFirst.shape[1]

	tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
	tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

	tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	result = net(tenPreprocessedFirst, tenPreprocessedSecond)

	multiplication = (intHeight/result.shape[2]) * (intWidth/result.shape[3])

	tenFlow = multiplication * torch.nn.functional.interpolate(input=result, size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	return tenFlow[0, :, :, :].cpu()

if __name__ == '__main__':
	netNetwork = model.PWCNet()
	netNetwork.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(
		__file__.replace('main.py', 'network-' + arguments_strModel + '.pytorch')).items()})
	netNetwork = netNetwork.cuda().eval()

	videos = tub.load_all_videos('../data/TUBCrowdFlow')
	frame_pairs = videos[0].get_frame_pairs()
	frame_pairs = frame_pairs[0:50]

	timer = utils.sTimer('total loading')
	for i, pair in enumerate(frame_pairs):
		tenFirst = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(pair.get_frames(0).get_image_path()))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
		tenSecond = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(pair.get_frames(1).get_image_path()))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

		tenOutput = estimate(netNetwork, tenFirst, tenSecond)
		flow_np = tenOutput.numpy().transpose(1, 2, 0)

		print(flow_np.max())

		#rgb = utils.flo_to_color(flow_np)
		#plt.imsave('results/with_flow_{}.png'.format(i), rgb)

	timer.show()
# end