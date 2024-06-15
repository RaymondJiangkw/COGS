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
import os
import json
from dnnlib import EasyDict
import torch
from torch.nn import functional as F
from utils.loss_utils import norm_xy, predict_hierarchical_correspondence
from gaussian_renderer import renderApprSurface
import sys
import numpy as np
from scene import Scene, GaussianModel
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from matplotlib import pyplot as plt
from argparse import Namespace
from utils.eval_pose import compute_rpe, compute_ATE
def read_cfg(path: str):
    assert os.path.exists(os.path.join(path, 'cfg_args'))
    with open(os.path.join(path, 'cfg_args')) as f:
        string = f.read()
    args = eval(string)
    return EasyDict(**vars(args).copy())

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

import numpy as np
from PIL import Image
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

def register(dataset, opt, pipe, load_iteration, eval_pose):
    load_iteration = dataset.load_iteration
    dataset = read_cfg(dataset.model_path)
    dataset.load_iteration = load_iteration
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration)
    gaussians.requires_grad_(False)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    training_data = scene.getTrainCameras()
    testing_data = scene.getTestCameras()

    # Evaluate pose accuracy
    if scene.scene_info.w_gt and eval_pose:
        W2C = []
        ref_W2C = []
        for c in training_data:
            W2C.append(c.world_view_transform.T.cpu().numpy())
            ref_W2C.append(c.ref_world_view_transform.T.cpu().numpy())
        
        W2C = np.stack(W2C)
        ref_W2C = np.stack(ref_W2C)
        rpe = compute_rpe(ref_W2C, W2C)
        ATE = compute_ATE(ref_W2C, W2C)
        print("Training Views: ")
        print("RPE:", rpe)
        print("ATE:", ATE)
        with open(os.path.join(scene.model_path, f"camera/iteration_{scene.loaded_iter}/pose.json"), 'w') as f:
            json.dump({"rpe": list(rpe), "ate": float(ATE)}, f)

    fused_data = sorted(
        list(map(lambda x: ('training', x), training_data)) + \
        list(map(lambda x: ('testing', x), testing_data)), key=lambda x: x[1].image_name)
    testing_idx_to_fused_idx = {}
    for testing_ptr in range(len(testing_data)):
        testing_idx_to_fused_idx[testing_ptr] = list(map(lambda x: x[1].image_name, fused_data)).index(testing_data[testing_ptr].image_name)
    
    for test_idx, test_view in tqdm(enumerate(testing_data), total=len(testing_data)):
        # Initialize from previous camera
        test_view.init_(fused_data[testing_idx_to_fused_idx[test_idx] - 1][1])
        test_view.cam_requires_grad_(True)
        optimizer = torch.optim.Adam([
            {"params": [test_view.quaternion], "lr": 1e-3}, 
            {"params": [test_view.T], "lr": 1e-2} 
        ], lr=0.0, maximize=False)
        # psnr_s = []
        for _ in range(200):
            optimizer.zero_grad()
            test_out = renderApprSurface(test_view, gaussians, pipe, background)
            
            kp0, kp1 = predict_hierarchical_correspondence(
                test_view.image, 
                test_out["render"].clamp(0., 1.), 
                threshold=0.5
            )

            xy0 = kp0 / 2 + .5
            xy1 = F.grid_sample(norm_xy(test_out["xy"])[None], 
                    kp1[None, None], mode='bilinear', align_corners=False).reshape(2, -1).permute(1, 0)
            
            loss = torch.nn.L1Loss()(test_view.image, test_out["render"].clamp(0, 1)) + \
                1e2 * torch.nn.L1Loss()(xy0, xy1)
            
            loss.backward()
            optimizer.step()
            # psnr_s.append(psnr(test_view.image, test_out["render"].clamp(0., 1.)).mean().detach().item())
        test_view.cam_requires_grad_(False)
        optimizer.zero_grad()
        # plt.plot(psnr_s)
        # plt.show()
        with torch.no_grad():
            test_out = renderApprSurface(test_view, gaussians, pipe, background)
            tqdm.write(f"{test_view.uid}: {psnr(test_view.image, test_out['render'].clamp(0., 1.)).mean()}")
            # plt.imshow(np.array(render_tensor([test_view.image, test_out['render'].clamp(0., 1.)])))
            # plt.show()
    scene.save(scene.loaded_iter, skip_test=False)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluating script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--eval_pose", action='store_true', help="Evaluate the pose metrics.")
    args = parser.parse_args(sys.argv[1:])
    print("Registering testing views of " + args.model_path)
    register(lp.extract(args), op.extract(args), pp.extract(args), args.load_iteration, args.eval_pose)

    # All done
    print("Register complete.")
