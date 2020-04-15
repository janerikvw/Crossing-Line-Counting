import numpy as np
import torch

import datasets
import LOI
from CSRNet.model import CSRNet

def pixelwise_loi(counting_result, flow_result):
    return


def regionwise_loi(counting_result, flow_result):
    return


# Initialize the crowd counting model
def init_cc_model():
    print("--- LOADING CSRNET ---")
    print("- Initialize architecture")
    cc_model = CSRNet()
    print("- Architecture to GPU")
    cc_model = cc_model.cuda()
    print("- Loading weights")
    checkpoint = torch.load('CSRNet/fudan_only_model_best.pth.tar')
    print("- Weights to GPU")
    cc_model.load_state_dict(checkpoint['state_dict'])
    print("--- DONE ---")
    return cc_model


# Run the crowd counting model on frame A
def run_cc_model(cc_model, frame):
    img = 255.0 * F.to_tensor(Image.open(frame.get_image_path()).convert('RGB'))
    img[0, :, :] = img[0, :, :] - 92.8207477031
    img[1, :, :] = img[1, :, :] - 95.2757037428
    img[2, :, :] = img[2, :, :] - 104.877445883
    img = img.cuda()
    cc_output = cc_model(img.unsqueeze(0))
    return cc_output


# Initialize the flow estimation model
def init_fe_model():
    return


def run_fe_model(fe_model, pair):
    return