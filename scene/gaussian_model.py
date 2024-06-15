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
from typing import List, Literal, Union
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, construct_rays_by_uv, construct_unit_dirs_by_uv, construct_uv, get_appr_ellipsoid_scaling, get_appr_ellipsoid_scaling_inverse, quaternion_multiply
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.cameras import DifferentiableCamera

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def requires_grad_(self, requires_grad=True):
        self._xyz.requires_grad_(requires_grad)
        self._features_dc.requires_grad_(requires_grad)
        self._features_rest.requires_grad_(requires_grad)
        self._scaling.requires_grad_(requires_grad)
        self._rotation.requires_grad_(requires_grad)
        self._opacity.requires_grad_(requires_grad)

    def training_setup(self, training_args, optimizable_cam_s = []):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        l += [
            {'params': [cam.quaternion for cam in optimizable_cam_s], 'lr': training_args.rotation_lr_init, 'name': f'cam_rotation'}, 
            {'params': [cam.T for cam in optimizable_cam_s], 'lr': training_args.translation_lr_init, 'name': f'cam_translation'}, 
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.cam_rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr_init, 
                                                             lr_final=training_args.rotation_lr_final, 
                                                             max_steps=training_args.camera_lr_max_steps)
        self.cam_translation_scheduler_args = get_expon_lr_func(lr_init=training_args.translation_lr_init, 
                                                                lr_final=training_args.translation_lr_final, 
                                                                max_steps=training_args.camera_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "cam_rotation":
                lr = self.cam_rotation_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "cam_translation":
                lr = self.cam_translation_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if not group["name"] in ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"]: continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if not group["name"] in tensors_dict: continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0], ), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            pass
            # big_points_vs = self.max_radii2D > max_screen_size
            # big_points_ws = self.get_scaling.max(dim=1).values > extent
            # prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

class BindableGaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
    
    class RelativeGaussianSet:
        def __init__(self, 
            cam: DifferentiableCamera, 
            # Only keep certain percent of pixels for back-projection
            retain_percent: float, 
            # Mask out certain area of pixels
            impose_mask: torch.Tensor, 
            # Initial opacity
            init_opacity: float, 
            # Mode of initial scaling: distance: determined by distance; small: small scaling
            init_scale: Union[Literal['distance'], Literal['small']], 
            # Whether to break the relationship between gaussians and the camera. Note that, 
            # the positions are parameterized by the depth regardless of this option.
            break_connection = False, 
            # Downsample the number of pixels
            scale_factor: int = 1, 
        ):
            sh_degree = 0
            self.init_scale = init_scale
            self.cam = cam
            self.break_connection = break_connection
            
            self.cam_view_world_transform = None if not break_connection else self.cam.view_world_transform.detach().clone()
            self.cam_intrinsics = None if not break_connection else self.cam.intrinsics.detach().clone()
            self.cam_quaternion = None if not break_connection else self.cam.quaternion.detach().clone()

            # Back-project pixels to gaussians
            self.h, self.w = cam.image.shape[-2:]
            self.h, self.w = self.h // scale_factor, self.w // scale_factor
            uv = torch.stack(torch.meshgrid(
                torch.arange(self.h, dtype=torch.float32, device=self.cam.load_device) * (1./self.h) + (0.5/self.h), 
                torch.arange(self.w, dtype=torch.float32, device=self.cam.load_device) * (1./self.w) + (0.5/self.w), 
            indexing='ij'))
            self.uv = uv.flip(0).reshape(2, -1).transpose(1, 0) # (N, 2)
            self.uv_mask = torch.rand_like(self.uv[:, 0]) < retain_percent # (N, )
            if impose_mask is not None:
                impose_mask = F.interpolate(impose_mask[None].float(), (self.h, self.w), mode='nearest')
                self.uv_mask = torch.logical_and(self.uv_mask, impose_mask.reshape(-1).bool())
            
            # Radiance
            image = F.interpolate(cam.image[None], (self.h, self.w), mode='bilinear', align_corners=False, antialias=True)
            fused_color = RGB2SH(image.reshape(3, -1).permute(1, 0))
            features = torch.zeros((fused_color.shape[0], 3, (sh_degree + 1) ** 2), dtype=fused_color.dtype, device=fused_color.device)
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
            self._features_dc = nn.Parameter(features[:, :, :1].transpose(1, 2)[self.uv_mask].contiguous())
            self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2)[self.uv_mask].contiguous())

            # Scaling
            if init_scale == 'small':
                self._scaling = nn.Parameter(torch.log(1e-3 * torch.ones((self._features_dc.size(0), 3), dtype=self._features_dc.dtype, device=self._features_dc.device)))
                self.scaling_coefficient = 1.
            elif init_scale == 'distance':
                # See Supplementary for derivation
                canonical_ray_dirs = construct_unit_dirs_by_uv(self.uv[self.uv_mask], self.cam.intrinsics.T)
                maximum_cosine = torch.zeros_like(canonical_ray_dirs[:, 0])
                for dh in [-1., 0., 1.]:
                    for dw in [-1., 0., 1.]:
                        if dh != 0. or dw != 0.:
                            delta_uv = construct_uv(self.h, self.w, self.cam.load_device, dh, dw)
                            unit_ray_dirs = construct_unit_dirs_by_uv(delta_uv[self.uv_mask], self.cam.intrinsics.T) # (N, 3)
                            maximum_cosine = torch.maximum(maximum_cosine, (canonical_ray_dirs * unit_ray_dirs).sum(dim=-1).abs())
                minimum_sine = (1. - maximum_cosine.clamp_max_(0.999999) ** 2).sqrt()[:, None] # (N, 1)
                self.scaling_coefficient = (minimum_sine / (1 - minimum_sine))

            # Rotation
            rots = torch.zeros((self._features_dc.size(0), 4), dtype=self._features_dc.dtype, device=self._features_dc.device)
            rots[:, 0] = 1
            self._rotation = nn.Parameter(rots)

            # Opacity
            self._opacity = nn.Parameter(inverse_sigmoid(init_opacity * torch.ones((self._features_dc.size(0), 1), dtype=self._features_dc.dtype, device=self._features_dc.device)))
        
        def requires_grad_(self, requires_grad=True):
            self._features_dc.requires_grad_(requires_grad)
            self._features_rest.requires_grad_(requires_grad)
            if getattr(self, '_scaling', None) is not None: self._scaling.requires_grad_(requires_grad)
            self._rotation.requires_grad_(requires_grad)
            self._opacity.requires_grad_(requires_grad)
        
        @property
        def rotation(self):
            return quaternion_multiply(self._rotation, self.cam_quaternion if self.break_connection else self.cam.quaternion)

        @property
        def scaling(self):
            depth = F.interpolate(self.cam.depth[None], (self.h, self.w), mode='bicubic', align_corners=False)
            depth = depth.reshape(-1, 1)[self.uv_mask] # (N, 1)
            scaling = depth * self.scaling_coefficient
            return (get_appr_ellipsoid_scaling_inverse(scaling) - 1e-3).clamp_min_(1e-6).repeat(1, 3)
        
        @torch.no_grad()
        def export(self, farest_percent = 1.):
            ray_origins, ray_dirs = construct_rays_by_uv((self.cam_view_world_transform if self.break_connection else self.cam.view_world_transform).T[None, ...], (self.cam_intrinsics if self.break_connection else self.cam.intrinsics).T[None, ...], self.uv)
            xyz, mask = self.cam.farest_point_sampling(ray_origins, ray_dirs, self.uv_mask, farest_percent)
            return xyz, SH2RGB(self._features_dc[mask]).squeeze(1)

        @property
        def xyz(self) -> torch.Tensor:
            ray_origins, ray_dirs = construct_rays_by_uv((self.cam_view_world_transform if self.break_connection else self.cam.view_world_transform).T[None, ...], (self.cam_intrinsics if self.break_connection else self.cam.intrinsics).T[None, ...], self.uv)
            ray_dirs = -ray_dirs
            
            depth = F.interpolate(self.cam.depth[None], (self.h, self.w), mode='bicubic', align_corners=False)
            depth = depth.reshape(-1, 1)
            ray_dirs = ray_dirs[self.uv_mask]
            depth = depth[self.uv_mask]

            scales = self.scaling
            rotations = torch.nn.functional.normalize(self._rotation)

            ellipsoid_scale = get_appr_ellipsoid_scaling(scales)
            ellipsoid_rotation = build_rotation(rotations).permute(0, 2, 1) # (N, 3, 3)
            local_ray_direction = torch.matmul(ray_dirs[:, None, :], ellipsoid_rotation).squeeze(-2) # (N, 3)
            correction = 1. / torch.linalg.vector_norm(local_ray_direction / ellipsoid_scale, ord=2, dim=-1)

            return ray_origins + ray_dirs * (depth + correction[:, None])


    def __init__(self):
        self.active_sh_degree = 0
        self.max_sh_degree = 0
        self.pc_s = []
        self.setup_functions()

    @property
    def get_scaling(self):
        return torch.cat([pc.scaling for pc in self.pc_s])
    
    @property
    def get_rotation(self):
        return self.rotation_activation(torch.cat([pc.rotation for pc in self.pc_s]))
    
    @property
    def get_xyz(self):
        return torch.cat([pc.xyz for pc in self.pc_s])
    
    @property
    def get_features(self):
        features_dc = torch.cat([pc._features_dc for pc in self.pc_s])
        features_rest = torch.cat([pc._features_rest for pc in self.pc_s])
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(torch.cat([pc._opacity for pc in self.pc_s]))
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    @torch.no_grad()
    def export(self, farest_percent = 1., retain_percent = 1.):
        xyz_s, rgb_s = [], []
        for pc in self.pc_s:
            xyz, rgb = pc.export(farest_percent)
            xyz_s.append(xyz)
            rgb_s.append(rgb)
        xyz_s = torch.cat(xyz_s)
        rgb_s = torch.cat(rgb_s)
        mask = torch.rand_like(xyz_s[:, 0]) < retain_percent
        xyz_s = xyz_s[mask]
        rgb_s = rgb_s[mask]
        return BasicPointCloud(
            points=xyz_s.cpu().numpy(), 
            colors=rgb_s.cpu().numpy(), 
            normals=None
        )

    def add_frame(self, cam: DifferentiableCamera, **kwargs):
        cam.registered = True
        self.pc_s.append(self.RelativeGaussianSet(
            cam, **kwargs
        ))
    
    def requires_grad_(self, requires_grad):
        for pc in self.pc_s:
            pc.requires_grad_(requires_grad)