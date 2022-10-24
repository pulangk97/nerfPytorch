import torch
import numpy as np
from piqa import SSIM
def MSEloss(x_in,x_gt):
    return torch.mean(torch.square(x_in-x_gt))


def L1loss(x_in,x_gt):
    return torch.mean(torch.abs(x_in-x_gt))


def TVloss(x_in,x_gt):
    def compute_tv_4d(field):
        dx = field[..., :, 1:] - field[..., :, :-1]
        dy = field[..., 1:, :] - field[..., :-1, :]
        return dx, dy

        # compute total variation loss
    x_in_dx, x_in_dy   = compute_tv_4d(x_in)
    x_out_dx, x_out_dy = compute_tv_4d(x_gt)
    tv_loss=0.5*torch.mean(torch.abs(x_in_dx-x_out_dx))+0.5*torch.mean(torch.abs(x_in_dy-x_out_dy))
    return tv_loss


def computPSNR(x_in,x_gt):
    return 10*torch.log10(torch.square(torch.max(x_gt))/MSEloss(x_in,x_gt))


## [N C H W]
def computSSIM(x_in,x_gt):
    ssim=SSIM()
    return ssim(x_in/torch.max(x_in),x_gt/torch.max(x_gt))