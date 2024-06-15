import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
from dnnlib import EasyDict
from torch import nn
from demo.predictor import VisualizationDemo

from PIL import Image
import numpy as np
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from fcclip import add_maskformer2_config, add_fcclip_config

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_fcclip_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

class MaskPredictor:
    def __init__(self, confidence_threshold=.5):
        cfg = setup_cfg(EasyDict(config_file=os.path.join(os.path.dirname(__file__), "configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml"), confidence_threshold=confidence_threshold, opts=['MODEL.WEIGHTS', os.path.join(os.path.dirname(__file__), "fcclip_cocopan.pth")]))

        self.demo = VisualizationDemo(cfg)
    
    def __call__(self, img: torch.Tensor):
        if isinstance(img, torch.Tensor):
            img = (img * 255.).permute(1, 2, 0).to(torch.uint8).cpu().numpy()[:, :, ::-1]
        elif isinstance(img, Image.Image):
            img = np.array(img)[:, :, ::-1]
        return self.demo.run_on_image(img)
