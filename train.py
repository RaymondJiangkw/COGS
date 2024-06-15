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
import torch
import numpy as np
import random as r
from random import randint
from dnnlib.util import EasyDict
from utils.graphics_utils import render_tensor
from utils.loss_utils import l1_loss, norm_xy, delta_xy, predict_hierarchical_correspondence, ssim
from gaussian_renderer import render, renderApprSurface
import sys
from scene import Scene, SceneFromScratch, GaussianModel, BindableGaussianModel
from utils.general_utils import safe_state
from time import gmtime, strftime
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

def get_radius(cameras):
    cam_centers = np.hstack([camera.camera_center.squeeze().detach().cpu().numpy()[:, None] for camera in cameras])
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    return diagonal * 1.1

def toggle_cam_grad(train_cameras, requires_grad, first_idx = 1):
    first_idx = min(max(first_idx, 0), len(train_cameras))
    for cam in train_cameras[:first_idx]:
        cam.quaternion.requires_grad_(False)
        cam.T.requires_grad_(False)
    for cam in train_cameras[first_idx:]:
        cam.quaternion.requires_grad_(requires_grad)
        cam.T.requires_grad_(requires_grad)

def toggle_depth_grad(train_cameras, requires_grad, first_idx = 0):
    first_idx = min(max(first_idx, 0), len(train_cameras))
    for cam in train_cameras[:first_idx]:
        cam.scales_and_shifts_requires_grad_(False)
    for cam in train_cameras[first_idx:]:
        cam.scales_and_shifts_requires_grad_(requires_grad)

def parameters_cam_quaternion(train_cameras):
    return [cam.quaternion for cam in train_cameras]

def parameters_cam_T(train_cameras):
    return [cam.T for cam in train_cameras]

def parameters_cam_scale(train_cameras):
    return sum([cam.parameters_scale for cam in train_cameras], [])

def parameters_cam_shift(train_cameras):
    return sum([cam.parameters_shift for cam in train_cameras], [])

def construct_coarse_solution(tb_writer: SummaryWriter, dataset, opt, pipe):
    gaussians = BindableGaussianModel()
    scene = SceneFromScratch(dataset, gaussians)

    train_cameras = scene.getTrainCameras()
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    add_opts = EasyDict(retain_percent=1., init_opacity=opt.init_opacity, init_scale=opt.init_scale, break_connection=opt.break_connection, scale_factor=opt.add_scale_factor)

    for view in train_cameras:
        view.set_up_scale_and_shift()

    # Add the first frame
    gaussians.add_frame(train_cameras[0], impose_mask=None, **add_opts)
    gaussians.requires_grad_(False)
    train_cameras[0].cam_requires_grad_(False)
    train_cameras[0].scales_and_shifts_requires_grad_(False)
    if TENSORBOARD_FOUND:
        with torch.no_grad():
            render_pkg = renderApprSurface(train_cameras[0], gaussians, pipe, background)
            tb_writer.add_images(f"{train_cameras[0].uid}/gt", train_cameras[0].image[None], 0)
            tb_writer.add_images(f"{train_cameras[0].uid}/render", render_pkg["render"].clamp(0., 1.)[None], 0)

    for idx in tqdm(range(1, len(train_cameras))):
        prev_cam = train_cameras[idx - 1]
        next_cam = train_cameras[idx]

        # 1. Search for the best camera location of current camera
        toggle_cam_grad(train_cameras, False)
        toggle_depth_grad(train_cameras, False)
        next_cam.init_(prev_cam)
        next_cam.cam_requires_grad_(True)
        next_cam.scales_and_shifts_requires_grad_(False)

        optimizer = torch.optim.Adam([
            {"params": [ next_cam.quaternion ], "lr": opt.rotation_finetune_lr}, 
            {"params": [ next_cam.T ], "lr": opt.translation_finetune_lr}
        ], lr=0.0)

        optimizer.zero_grad()
        for step in tqdm(range(opt.register_steps), desc=f'Camera Search for {idx}-th Camera'):
            render_pkg = renderApprSurface(next_cam, gaussians, pipe, background)
            
            kp0, kp1, conf = predict_hierarchical_correspondence(
                next_cam.image, 
                render_pkg["render"].clamp(0., 1.), 
                threshold=opt.correspondence_threshold, 
                returns_conf=True
            )
            
            xy0 = kp0 / 2 + .5
            xy1 = torch.nn.functional.grid_sample(norm_xy(render_pkg["xy"])[None], kp1[None, None], mode='bilinear', align_corners=True).reshape(2, -1).permute(1, 0)
            mask = torch.logical_and(xy1 > 0., xy1 < 1.).all(dim=-1)
            xy0, xy1, conf = xy0[mask], xy1[mask], conf[mask]
            
            loss_2d = torch.nn.L1Loss()(xy0, xy1)
            # loss_2d = ((xy0 - xy1) * conf[:, None]).abs().mean()
            loss_l1 = torch.nn.L1Loss()(render_pkg["render"], next_cam.image)
            loss = opt.loss_2d_correspondence_weight * loss_2d + opt.loss_rgb_correspondence_weight * loss_l1

            if TENSORBOARD_FOUND:
                tb_writer.add_scalar(f"{next_cam.uid}/correspondence_loss", loss_2d, step)
                tb_writer.add_scalar(f"{next_cam.uid}/rgb_loss", loss_l1, step)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
        if TENSORBOARD_FOUND:
            with torch.no_grad():
                render_pkg = renderApprSurface(next_cam, gaussians, pipe, background)
                tb_writer.add_images(f"{next_cam.uid}/gt", next_cam.image[None], 0)
                tb_writer.add_images(f"{next_cam.uid}/render", render_pkg["render"].clamp(0., 1.)[None], 0)
        
        # 2. Alignment
        if idx % opt.add_frame_interval == 0 or idx == len(train_cameras) - 1:
            toggle_cam_grad(train_cameras, True)
            toggle_depth_grad(train_cameras, True)

            optimizer = torch.optim.Adam([
                {"params": parameters_cam_quaternion(train_cameras[1:]), "lr": opt.rotation_finetune_lr}, 
                {"params": parameters_cam_T(train_cameras[1:]), "lr": opt.translation_finetune_lr}, 
                {"params": parameters_cam_scale(train_cameras), "lr": opt.scale_finetune_lr}, 
                {"params": parameters_cam_shift(train_cameras), "lr": opt.shift_finetune_lr}
            ], lr=0.0)

            for step in tqdm(range(opt.align_steps), desc=f'Alignment Tuning for {idx}-th Camera'):
                if step % 2 == 0:
                    render_pkg = renderApprSurface(next_cam, gaussians, pipe, background)
                    
                    kp0, kp1, conf = predict_hierarchical_correspondence(
                        next_cam.image, 
                        render_pkg["render"].clamp(0., 1.), 
                        threshold=opt.correspondence_threshold, 
                        returns_conf=True
                    )
                    
                    xy0 = kp0 / 2 + .5
                    xy1 = torch.nn.functional.grid_sample(norm_xy(render_pkg["xy"])[None], kp1[None, None], mode='bilinear', align_corners=True).reshape(2, -1).permute(1, 0)
                    mask = torch.logical_and(xy1 > 0., xy1 < 1.).all(dim=-1)
                    xy0, xy1, conf = xy0[mask], xy1[mask], conf[mask]
                    
                    hole_mask = delta_xy(render_pkg["xy"]).detach().squeeze()[None, None]
                    
                    depth0 = torch.nn.functional.grid_sample(next_cam.depth[None] * (~hole_mask), kp0[None, None], mode='bilinear', align_corners=False).squeeze()
                    
                    depth1 = torch.nn.functional.grid_sample(render_pkg["depth"][None] * (~hole_mask), kp1[None, None], mode='bilinear', align_corners=False).squeeze()

                    loss_2d = torch.nn.L1Loss()(xy0, xy1)
                    # loss_2d = ((xy0 - xy1) * conf[:, None]).abs().mean()
                    loss_l1 = torch.nn.L1Loss()(render_pkg["render"], next_cam.image)
                    loss_depth = torch.nn.L1Loss()(depth0, depth1.detach())
                    
                    loss = opt.loss_2d_correspondence_weight * loss_2d + opt.loss_rgb_correspondence_weight * loss_l1 + opt.loss_depth_correspondence_weight * loss_depth

                    if TENSORBOARD_FOUND:
                        tb_writer.add_scalar(f"{next_cam.uid}/correspondence_loss", loss_2d, step + opt.register_steps)
                        tb_writer.add_scalar(f"{next_cam.uid}/rgb_loss", loss_l1, step + opt.register_steps)
                        tb_writer.add_scalar(f"{next_cam.uid}/depth_loss", loss_depth, step + opt.register_steps)
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    # Maintain Previous Results
                    random_cam = r.choice(train_cameras[:idx])
                    render_pkg = renderApprSurface(random_cam, gaussians, pipe, background)
                    
                    kp0, kp1, conf = predict_hierarchical_correspondence(
                        random_cam.image, 
                        render_pkg["render"].clamp(0., 1.), 
                        threshold=opt.correspondence_threshold, 
                        returns_conf=True
                    )
                    
                    xy0 = kp0 / 2 + .5
                    xy1 = torch.nn.functional.grid_sample(norm_xy(render_pkg["xy"])[None], kp1[None, None], mode='bilinear', align_corners=True).reshape(2, -1).permute(1, 0)
                    mask = torch.logical_and(xy1 > 0., xy1 < 1.).all(dim=-1)
                    xy0, xy1, conf = xy0[mask], xy1[mask], conf[mask]

                    loss_2d = torch.nn.L1Loss()(xy0, xy1)
                    # loss_2d = ((xy0 - xy1) * conf[:, None]).abs().mean()
                    loss_l1 = torch.nn.L1Loss()(render_pkg["render"], random_cam.image)
                    
                    loss = opt.loss_2d_correspondence_weight * loss_2d + opt.loss_rgb_correspondence_weight * loss_l1

                    if TENSORBOARD_FOUND:
                        tb_writer.add_scalar(f"{random_cam.uid}/{next_cam.uid}_correspondence_loss", loss_2d, step)
                        tb_writer.add_scalar(f"{random_cam.uid}/{next_cam.uid}_rgb_loss", loss_l1, step)
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            
            if TENSORBOARD_FOUND:
                with torch.no_grad():
                    render_pkg = renderApprSurface(next_cam, gaussians, pipe, background)
                    tb_writer.add_images(f"{next_cam.uid}/render", render_pkg["render"].clamp(0., 1.)[None], 1)
            
                
        # 3. Done
        toggle_cam_grad(train_cameras, False)
        toggle_depth_grad(train_cameras, False)

        if idx % opt.add_frame_interval == 0 or idx == len(train_cameras) - 1:
            with torch.no_grad():
                render_pkg = renderApprSurface(next_cam, gaussians, pipe, background)
                # Only project to previously empty regions
                impose_mask = delta_xy(render_pkg["xy"]).detach().squeeze()[None]
                impose_mask = torch.logical_and(impose_mask, render_pkg["depth"] <= 1e-3)

                hole_mask = delta_xy(render_pkg["xy"]).detach().squeeze()[None, None]
                depth_diff_mask = ((render_pkg["depth"].squeeze() - next_cam.depth.squeeze()) * (~hole_mask.squeeze())).squeeze().reshape_as(impose_mask)
                
                # Additionally, project to regions which are unseen in previous frames, 
                # but overlap with certain observed regions.
                impose_mask = torch.logical_or(impose_mask, depth_diff_mask > opt.depth_diff_tolerance)
                # Dilate impose_mask
                impose_mask = (torch.nn.functional.conv2d(impose_mask.float()[None], torch.ones(1, 1, opt.dilate_kernel_size, opt.dilate_kernel_size, device=impose_mask.device), padding='same') == opt.dilate_kernel_size ** 2)[0]
                gaussians.add_frame(next_cam, **add_opts, impose_mask=impose_mask)
                gaussians.requires_grad_(False)
                if TENSORBOARD_FOUND:
                    tb_writer.add_images(f"{next_cam.uid}/add_mask", impose_mask.squeeze().float()[None].expand(3, -1, -1)[None], 2)
                    render_pkg = renderApprSurface(next_cam, gaussians, pipe, background)
                    tb_writer.add_images(f"{next_cam.uid}/render", render_pkg["render"].clamp(0., 1.)[None], 2)
                    tb_writer.add_images(f"{next_cam.uid}/depth_diff", (((render_pkg["depth"].squeeze() - next_cam.depth.squeeze()) * (~hole_mask.squeeze())).squeeze().abs().clamp(0., 1e1)[None, ...].expand(3, -1, -1) / 1e1)[None], 2)
    
    # Save Intermediate Result for Debugging
    if TENSORBOARD_FOUND:
        with torch.no_grad():
            for cam in train_cameras:
                render_pkg = renderApprSurface(cam, gaussians, pipe, background)
                tb_writer.add_images(f"final/{cam.uid}_gt", cam.image[None], 0)
                tb_writer.add_images(f"final/{cam.uid}_render", render_pkg["render"].clamp(0., 1.)[None], 0)
    
    return scene.train_cameras, scene.test_cameras, gaussians.export(opt.farest_percent, opt.retain_percent), get_radius(train_cameras)

def refinement(tb_writer, train_cams, test_cams, init_pc, radius, dataset, opt, pipe, testing_interval, saving_interval):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, train_cameras_override=train_cams, test_cameras_override=test_cams, cameras_extent=radius, init_pc=init_pc)
    gaussians.spatial_lr_scale = 5.

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if TENSORBOARD_FOUND:
        with torch.no_grad():
            for cam in scene.getTrainCameras():
                render_pkg = renderApprSurface(cam, gaussians, pipe, background)
                tb_writer.add_images(f"final/{cam.uid}_render", render_pkg["render"].clamp(0., 1.)[None], 1)
    toggle_cam_grad(scene.getTrainCameras(), True)
    gaussians.training_setup(opt, scene.getTrainCameras()[1:])

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    scene_iteration, iteration = 0, first_iter
    
    last_scene_iter_upsample = 0
    last_scene_iter_progress = 0
    last_scene_iter_saving   = 0
    last_scene_iter_densify  = 0
    last_scene_iter_reset_o  = 0
    sum_cam_optim            = 0
    sum_scene_optim          = 0
    while scene_iteration < opt.iterations:
        iter_start.record()

        gaussians.update_learning_rate(scene_iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if scene_iteration % 1000 == 0 and scene_iteration != last_scene_iter_upsample:
            gaussians.oneupSHdegree()
            last_scene_iter_upsample = scene_iteration

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = list(range(len(scene.getTrainCameras())))
        
        viewpoint_cam_idx = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        viewpoint_cam = scene.getTrainCameras()[viewpoint_cam_idx]

        v = None # 0 - Cam Optimization | 1 - Scene Optimization
        if viewpoint_cam_idx == 0 or scene_iteration > opt.cam_optim_until_iter or scene_iteration < opt.cam_optim_from_iter:
            v = 1
        else:
            v = r.choice([0, 1])
        
        if v == 0: sum_cam_optim += 1
        else: sum_scene_optim += 1
        
        toggle_cam_grad(scene.getTrainCameras(), False)
        
        if v == 0:
            viewpoint_cam.cam_requires_grad_(True)
            gaussians.requires_grad_(False)
        elif v == 1:
            gaussians.requires_grad_(True)
            scene_iteration += 1

        bg = background

        render_pkg = renderApprSurface(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if scene_iteration % 10 == 0 and scene_iteration != last_scene_iter_progress:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
                last_scene_iter_progress = scene_iteration
            if scene_iteration == opt.iterations - 1:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_interval, scene, render, (pipe, background))

            if tb_writer:
                tb_writer.add_scalar("stats/scene_optim_sum", sum_scene_optim, iteration)
                tb_writer.add_scalar('stats/cam_optim_sum', sum_cam_optim, iteration)
            
            if (scene_iteration % saving_interval == 0) and scene_iteration != last_scene_iter_saving:
                print("\n[SCENE ITER {}] Saving Gaussians".format(scene_iteration))
                print("Saving Path:", dataset.model_path)
                scene.save(scene_iteration)
                last_scene_iter_saving = scene_iteration

            # Densification
            if scene_iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                if v == 1:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if scene_iteration > opt.densify_from_iter and scene_iteration % opt.densification_interval == 0 and scene_iteration != last_scene_iter_densify:
                    size_threshold = 20 if scene_iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    last_scene_iter_densify = scene_iteration
                
                if scene_iteration % opt.opacity_reset_interval == 0 and scene_iteration != last_scene_iter_reset_o:
                    gaussians.reset_opacity()
                    last_scene_iter_reset_o = scene_iteration

            # Optimizer step
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            
            iteration += 1

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_interval, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('stats/iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration % testing_interval == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : []}, # Cameras of test views are not registered yet.
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_interval:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('stats/total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def prepare_output_and_logger(args, opt):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = strftime("%Y-%m-%d_%H-%M-%S_", gmtime()) + os.path.basename(args.source_path) + "_" + str(args.num_images)
        args.model_path = os.path.join("./output/", unique_str)
    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    with open(os.path.join(args.model_path, "training_options.json"), 'w') as f:
        json.dump(vars(opt), f)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def render_depth(depth: torch.Tensor, inverse=False):
    if inverse:
        return (depth.max() - depth) / (depth.max() - depth.min() + 1e-5)
    else:
        return (depth - depth.min()) / (depth.max() - depth.min() + 1e-5)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--evaluate_interval", type=int, default=1_000)
    parser.add_argument("--save_interval", nargs="+", type=int, default=1_000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    # Initialize system state (RNG)
    safe_state(args.quiet)

    dataset = lp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)
    
    tb_writer = prepare_output_and_logger(dataset, opt)
    print("Optimizing " + args.model_path)
    train_cameras, test_cameras, init_pc, radius = construct_coarse_solution(tb_writer, dataset, opt, pipe)
    refinement(tb_writer, train_cameras, test_cameras, init_pc, radius, dataset, opt, pipe, args.evaluate_interval, args.save_interval)

    # All done
    print("\nTraining complete.")
