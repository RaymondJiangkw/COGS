#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchvision.transforms import Grayscale
from submodules.QuadTreeAttention.FeatureMatching import QuadTreeLoFTR

__correspondence_model = None

@torch.no_grad()
def predict_hierarchical_correspondence(prev_image, next_image, scales = [1.0, 0.5, 0.25], returns_conf=False, **kwargs):
    kp0, kp1, conf = [], [], []
    for scale in scales:
        p_i = torch.nn.functional.interpolate(prev_image[None], scale_factor=scale, mode='bilinear', align_corners=False, antialias=True)[0] if scale < 1. else prev_image
        n_i = torch.nn.functional.interpolate(next_image[None], scale_factor=scale, mode='bilinear', align_corners=False, antialias=True)[0] if scale < 1. else next_image
        r_kp0, r_kp1, r_conf = __predict_correspondence(p_i, n_i, **kwargs)
        kp0.append(r_kp0)
        kp1.append(r_kp1)
        conf.append(r_conf)
    if returns_conf:
        return torch.cat(kp0), torch.cat(kp1), torch.cat(conf)
    else:
        return torch.cat(kp0), torch.cat(kp1)

@torch.no_grad()
def __predict_correspondence(prev_image, next_image, threshold=0., normalize=True):
    global __correspondence_model
    if __correspondence_model is None: __correspondence_model = QuadTreeLoFTR().eval().requires_grad_(False).cuda()
    size = prev_image.shape[1:]
    matching = __correspondence_model({
        "image0": Grayscale()(prev_image[None, ...]), 
        "image1": Grayscale()(next_image[None, ...])
    })
    mask = matching['confidence'] > threshold
    keypoints0 = matching['keypoints0'][mask]
    keypoints1 = matching['keypoints1'][mask]

    if normalize:
        keypoints0[:, 0] = 2 * keypoints0[:, 0] / (size[1] - 1) - 1
        keypoints0[:, 1] = 2 * keypoints0[:, 1] / (size[0] - 1) - 1

        keypoints1[:, 0] = 2 * keypoints1[:, 0] / (size[1] - 1) - 1
        keypoints1[:, 1] = 2 * keypoints1[:, 1] / (size[0] - 1) - 1

    return keypoints0.contiguous(), keypoints1.contiguous(), matching['confidence'][mask]

def delta_xy(xy):
    identity = F.affine_grid(
        torch.eye(2, 3, device=xy.device)[None], xy[None].shape, align_corners=False)[0].permute(2, 0, 1) / 2 + .5
    identity[0] *= xy.shape[-1]
    identity[1] *= xy.shape[-2]
    identity = torch.floor(identity)
    return ((identity - xy).abs() > 1).any(dim=0)

def norm_xy(xy):
    identity = F.affine_grid(
        torch.eye(2, 3, device=xy.device)[None], xy[None].shape, align_corners=True)[0].permute(2, 0, 1) / 2 + .5
    xy = torch.stack((xy[0] / xy.shape[-1], xy[1] / xy.shape[-2]))
    # Even though we render the screen-space coordinates accurately, 
    # there still remains a problem that when we do bilinear sampling, 
    # holes could make the sampled coordinates inaccurate.
    # E.g., 
    # (x, y) --- (0, 0), 
    #   |     ?     |
    # (0, 0) --- (x+1, y+1)
    # Applying bilinear sampling results in (1/4) * (x, y) + (1/4) * (x+1, y+1).
    # Therefore, we need to fill in the holes.
    return xy + (identity - xy).detach() # Fill the holes

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

