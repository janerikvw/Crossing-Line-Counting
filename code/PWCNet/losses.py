import torch
import model_utils

def abs_robust_loss(diff, mask, q=0.4):
    diff = torch.pow(torch.abs(diff) + 0.01, q)
    diff = torch.mul(diff, mask)
    diff_sum = diff.sum()
    loss_mean = diff_sum / (mask.sum() * 2 + 1e-6)
    return loss_mean

# The photometric losses
def create_photometric_losses(frame1, frame2, flow_fw, flow_bw):
    # Could be highly optimized

    # 0/1 map with pixels which are occluded
    occ_fw, occ_bw = model_utils.occlusion(flow_fw, flow_bw)
    mask_fw = 1. - occ_fw
    mask_bw = 1. - occ_bw

    # Apply warp to move to the other frame
    img1_warp = model_utils.backwarp(frame1, flow_bw)
    img2_warp = model_utils.backwarp(frame2, flow_fw)

    # Calc photometric loss
    non_occ_photometric_loss = abs_robust_loss(frame1 - img2_warp, torch.ones_like(mask_fw)) + \
                            abs_robust_loss(frame2 - img1_warp, torch.ones_like(mask_bw))

    occ_photometric_loss = abs_robust_loss(frame1 - img2_warp, mask_fw) + \
                       abs_robust_loss(frame2 - img1_warp, mask_bw)

    return non_occ_photometric_loss, occ_photometric_loss