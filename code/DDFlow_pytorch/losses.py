import torch
import model_utils


# What exact loss is this? Look for better explaination
def abs_robust_loss(diff, mask, q=0.4):
    diff = torch.pow(torch.abs(diff) + 0.01, q)
    diff = torch.mul(diff, mask)
    diff_sum = diff.sum()
    loss_mean = diff_sum / (mask.sum() * 2 + 1e-6)
    return loss_mean

# The photometric losses
def create_photometric_losses(frame1, frame2, flow_fw, flow_bw):
    # Could be highly optimized
    losses = {}

    # 0/1 map with pixels which are occluded
    occ_fw, occ_bw = model_utils.occlusion(flow_fw, flow_bw)
    mask_fw = 1. - occ_fw
    mask_bw = 1. - occ_bw

    # Apply warp to move to the other frame
    img1_warp = model_utils.backwarp(frame1, flow_bw)
    img2_warp = model_utils.backwarp(frame2, flow_fw)

    # Calc photometric loss
    abs_losses = {}
    abs_losses['no_occlusion'] = abs_robust_loss(frame1 - img2_warp, torch.ones_like(mask_fw)) + \
                            abs_robust_loss(frame2 - img1_warp, torch.ones_like(mask_bw))

    abs_losses['occlusion'] = abs_robust_loss(frame1 - img2_warp, mask_fw) + \
                       abs_robust_loss(frame2 - img1_warp, mask_bw)

    losses['abs_robust_mean'] = abs_losses

    return losses


# The photometric losses
def create_distilled_losses(flow_fw, flow_bw, patch_flow_fw, patch_flow_bw):

    # 0/1 map with pixels which are occluded
    occ_fw, occ_bw = model_utils.occlusion(flow_fw, flow_bw)
    patch_occ_fw, patch_occ_bw = model_utils.occlusion(patch_flow_fw, patch_flow_bw)

    valid_mask_fw = torch.clamp(patch_occ_fw - occ_fw, 0., 1.)
    valid_mask_bw = torch.clamp(patch_occ_bw - occ_bw, 0., 1.)

    loss = (abs_robust_loss(flow_fw - patch_flow_fw, valid_mask_fw) +
            abs_robust_loss(flow_bw - patch_flow_bw, valid_mask_bw)) / 2

    return loss