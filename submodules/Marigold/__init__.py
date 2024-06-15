import os
import sys
sys.path.append(os.path.dirname(__file__))

import cv2
import torch
from torch import nn
from torch.nn import functional as F

from PIL import Image
import numpy as np
from src.model.marigold_pipeline import MarigoldPipeline
from src.util.ensemble import ensemble_depths
from src.util.image_util import resize_max_res
from torchvision.utils import make_grid

@torch.no_grad()
def render_tensor(img: torch.Tensor, normalize: bool = False, nrow: int = 8) -> Image.Image:
    def process_dtype(img):
        if img.dtype == torch.uint8:
            img = img.to(torch.float32) / 255.
            if normalize:
                img = img * 2 - 1
        return img
    if type(img) == list:
        img = torch.cat([process_dtype(i) if len(i.shape) == 4 else process_dtype(i[None, ...]) for i in img], dim=0).expand(-1, 3, -1, -1)
    elif len(img.shape) == 3:
        img = process_dtype(img).expand(3, -1, -1)
    elif len(img.shape) == 4:
        img = process_dtype(img).expand(-1, 3, -1, -1)
    
    img = img.squeeze()
    
    if normalize:
        img = img / 2 + .5
    
    if len(img.shape) == 3:
        return Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    elif len(img.shape) == 2:
        return Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8))
    elif len(img.shape) == 4:
        return Image.fromarray((make_grid(img, nrow=nrow).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

def resize_tensor_max_res(img: torch.Tensor, max_edge_resolution):
    original_height, original_width = img.shape[-2:]
    downscale_factor = min(max_edge_resolution / original_width, max_edge_resolution / original_height)
    
    new_height = int(original_height * downscale_factor)
    new_width = int(original_width * downscale_factor)
    
    if len(img.shape) == 3:
        img = img[None]
    
    resized_img = torch.nn.functional.interpolate(img, (new_height, new_width), 
                                                 mode='bilinear', align_corners=False, antialias=True)[0]
    return resized_img

class MariGold(nn.Module):
    def __init__(self, device: torch.device, n_repeat = 10, denoise_steps=10):
        super().__init__()

        self.checkpoint_path = "Bingxin/Marigold"
        self.resize_to_max_res = 768
        self.n_repeat = n_repeat
        self.denoise_steps = denoise_steps
        self.regularizer_strength = 0.02
        self.max_iter = 5
        self.tol = 1e-3
        self.device = device

        self.model = MarigoldPipeline.from_pretrained(self.checkpoint_path).eval().requires_grad_(False).to(device)
    @torch.no_grad()
    def forward(self, input_image: torch.Tensor, conditioning_depth: torch.Tensor = None, inv: bool = True, n_repeat: int = None):
        size = input_image.shape[-2:]
        rgb_norm = resize_tensor_max_res(input_image.clamp(0., 1.), self.resize_to_max_res)[None]
        
        init_depth_latent = None
        
        if conditioning_depth is not None:
            conditioning_depth = resize_tensor_max_res(conditioning_depth, self.resize_to_max_res)[None]
            init_depth_latent = self.model.encode_depth(conditioning_depth)

        depth_pred_ls = []
        for i_rep in range(n_repeat or self.n_repeat):
            depth_pred_raw = self.model.forward(
                rgb_norm, num_inference_steps=self.denoise_steps, init_depth_latent=init_depth_latent
            )
            # clip prediction
            depth_pred_raw = torch.clip(depth_pred_raw, -1.0, 1.0)
            depth_pred_ls.append(depth_pred_raw.detach().cpu().numpy().copy())

        depth_preds = np.concatenate(depth_pred_ls, axis=0).squeeze()

        # Test-time ensembling
        if (n_repeat or self.n_repeat) > 1:
            depth_pred, pred_uncert = ensemble_depths(
                depth_preds,
                regularizer_strength=self.regularizer_strength,
                max_iter=self.max_iter,
                tol=self.tol,
                reduction="median",
                max_res=None,
                device=self.device,
            )
        else:
            depth_pred = depth_preds
        
        depth_pred = cv2.resize(depth_pred, size[::-1])
        if not inv:
            return torch.from_numpy(depth_pred).to(torch.float32).to(self.device)
        # pred_img = Image.fromarray(depth_pred)
        # pred_img = pred_img.resize(input_image.size)
        # depth_pred = np.asarray(pred_img)
        depth_pred[depth_pred < 1e-8] = 1e-8
        return torch.from_numpy(1. / depth_pred).to(torch.float32).to(self.device)