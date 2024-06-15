import os
import sys
# sys.path.append(os.path.join())

import torch
from torch import nn
from torch.nn import functional as F

from .src.config.default import get_cfg_defaults
from .src.loftr import LoFTR
from .src.utils.misc import lower_config

class QuadTreeLoFTR(nn.Module):
    def __init__(self, setting = 'outdoor'):
        super().__init__()
        
        config = get_cfg_defaults()
        config.merge_from_file(os.path.join(os.path.dirname(__file__), f'./configs/loftr/{setting}/loftr_ds_quadtree.py'))
        config = lower_config(config)
        
        matcher = LoFTR(config=config["loftr"])
        state_dict = torch.load(os.path.join(os.path.dirname(__file__), f'../{setting}.ckpt'), map_location="cpu")["state_dict"]
        matcher.load_state_dict(state_dict, strict=True)
        
        self.new_shape = (480, 640)
        
        self.matcher = matcher
    def forward(self, x):
        image0, image1 = x['image0'], x['image1']
        original_shape = image0.shape[-2:]
        
        image0 = F.interpolate(image0, self.new_shape, mode='bilinear', align_corners=False, antialias=True)
        image1 = F.interpolate(image1, self.new_shape, mode='bilinear', align_corners=False, antialias=True)
        
        batch = {
            "image0": image0,
            "image1": image1,
        }
        
        self.matcher(batch)
        
        keypoints0 = batch["mkpts0_f"]
        keypoints1 = batch["mkpts1_f"]
        confidence = batch["mconf"]
        
        keypoints0[..., 0] *= original_shape[-1] / self.new_shape[-1]
        keypoints0[..., 1] *= original_shape[-2] / self.new_shape[-2]
        
        keypoints1[..., 0] *= original_shape[-1] / self.new_shape[-1]
        keypoints1[..., 1] *= original_shape[-2] / self.new_shape[-2]
        
        return { 'keypoints0': keypoints0, 'keypoints1': keypoints1, 'confidence': confidence }