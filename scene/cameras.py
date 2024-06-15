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
from torch import nn
from torch.nn import functional as F
import numpy as np
from utils.graphics_utils import construct_intrinsics, getProjectionMatrix2
from PIL import Image
import os

# This scaling is to scale the monocular depth predicted by MariGold 
# back to [0, 1]. For other sources of depths, you may also want to 
# scale it to [0, 1] to facilitate optimization.
BASE_SCALE = 150
BASE_SHIFT = 10

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

from pytorch3d.ops import sample_farthest_points

class DifferentiableCamera(nn.Module):
    def __init__(self, 
        # Property
        uid: int, image_name: str, load_device: torch.device, 
        width: float, height: float, 
        image_width: float, image_height: float, 
        image_path: str, depth_path: str, seg_mask_path: str, 
        # Extrinsics (init.)
        quaternionOrR: torch.Tensor, T: torch.Tensor, 
        # Intrinsics
        Focalx: float, Focaly: float, 
        Offsetx: float, Offsety: float, 
        # Options
        scale_and_shift_mode: str, 
        # Reference Extrinsics (facilitate evaluation)
        ref_quaternionOrR: torch.Tensor = None, ref_T: torch.Tensor = None, 
    ) -> None:
        super().__init__()
        assert scale_and_shift_mode in ['mask', 'whole']

        self.uid = uid
        self.image_name = image_name
        self.load_device = load_device
        self.width, self.height = width, height
        self.image_width, self.image_height = image_width, image_height
        self.registered = False
        self.optimized_iteration = 0

        self.image_path     = image_path
        self.depth_path     = depth_path if os.path.exists(depth_path) else None
        self.seg_mask_path  = seg_mask_path if os.path.exists(seg_mask_path) else None

        self.__image = None
        self.__depth = None
        self.__seg_mask = None

        register = lambda x, y: self.register_parameter(x, torch.nn.Parameter(y))

        # Extrinsics
        assert quaternionOrR.dtype == T.dtype
        quaternionOrR = quaternionOrR.reshape(-1)
        _quaternion = quaternionOrR.reshape(4, ) if len(quaternionOrR) == 4 else self.matrix_to_quaternion(quaternionOrR.reshape(3, 3)).reshape(4, )
        _T = T.reshape(3, )

        register('quaternion', _quaternion)
        register('T', _T)
        
        if ref_quaternionOrR is not None:
            self.register_buffer('ref_quaternion', ref_quaternionOrR.reshape(4, ) if len(ref_quaternionOrR) == 4 else self.matrix_to_quaternion(ref_quaternionOrR.reshape(3, 3)).reshape(4, ))
        if ref_T is not None:
            self.register_buffer('ref_T', ref_T.reshape(3, ))
        
        # Intrinsics
        register('Focalx', torch.tensor(float(Focalx)))
        register('Focaly', torch.tensor(float(Focaly)))
        register('Skew', torch.tensor(float(0.)))
        register('Offsetx', torch.tensor(float(Offsetx)))
        register('Offsety', torch.tensor(float(Offsety)))
        self.znear, self.zfar = 0.01, 100.0

        self.scale_and_shift_mode = scale_and_shift_mode
        self.scales_and_shifts = {}
    
    def __getattr__(self, name: str):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        return self.__getattribute__(name)
    
    @property
    def depth(self):
        if self.__depth is None:
            self.__depth = (torch.from_numpy(np.load(self.depth_path)).float().squeeze().to(self.load_device) - BASE_SHIFT) / BASE_SCALE
            self.__depth = F.interpolate(self.__depth[None, None, :, :], (self.image_height, self.image_width), mode='bicubic', align_corners=False)[0]
        if self.scale_and_shift_mode == 'mask':
            depth = torch.zeros_like(self.__depth)
            for mask_idx in self.scales_and_shifts:
                segmentation = self.seg_mask[mask_idx].bool().squeeze().unsqueeze(0)
                depth[segmentation] = torch.exp(self.scales_and_shifts[mask_idx][0]) * self.scales_and_shifts[mask_idx][2] * (self.__depth[segmentation] + self.scales_and_shifts[mask_idx][1] + self.scales_and_shifts[mask_idx][3])
            return depth * BASE_SCALE + BASE_SHIFT
        else:
            return torch.exp(self.scales_and_shifts[0]) * (self.__depth + self.scales_and_shifts[1]) * BASE_SCALE + BASE_SHIFT
    
    @torch.no_grad()
    def farest_point_sampling(self, ray_origins, ray_dirs, uv_mask, percent):
        depth = self.depth.squeeze(0)
        h, w = depth.shape
        ray_dirs = ray_dirs.reshape(h, w, 3)
        uv_mask = uv_mask.reshape(h, w)

        N = uv_mask.sum().item()
        idx_mask = torch.zeros_like(uv_mask).long()
        idx_mask[uv_mask] = torch.arange(N, device=uv_mask.device).long()

        selected_mask_s = []
        p_s = []

        for mask_idx in range(len(self.seg_mask)):
            # Shrink the semantic mask for better boundary
            semantic_mask = (self.seg_mask[mask_idx] > 0.).float()[None, None]
            semantic_mask = torch.nn.functional.conv2d(semantic_mask, torch.ones(1, 1, 5, 5, device=semantic_mask.device), padding='same') == 25
            segmentation = torch.logical_and(torch.nn.functional.interpolate(semantic_mask.float(), (h, w), mode='nearest').squeeze().bool(), uv_mask)
            if segmentation.sum().item() <= 0:
                continue
            pred_depth = torch.masked_select(depth, segmentation).reshape(-1, 1)
            mask_ray_dirs = ray_dirs[segmentation].reshape(-1, 3)
            points = ray_origins - mask_ray_dirs * pred_depth # (?, 3)
            if percent < 1 and int(len(points) * percent) > 2:
                p, idx = sample_farthest_points(points[None, ...], K=int(len(points) * percent))
                selected_mask_s.append(idx_mask[segmentation][idx[0]])
                p_s.append(p[0])
            else:
                selected_mask_s.append(idx_mask[segmentation])
                p_s.append(points)
        return torch.cat(p_s), torch.cat(selected_mask_s)
    
    @torch.no_grad()
    def set_up_scale_and_shift(self):
        self.depth # Trigger lazy loading
        depth = self.__depth.squeeze(0)
        if self.scale_and_shift_mode == 'mask':
            self.scales_and_shifts = {}
            for mask_idx in range(len(self.seg_mask)):
                segmentation = self.seg_mask[mask_idx].float()
                if segmentation.sum().item() <= 16:
                    # Skip too small regions
                    continue
                if ( segmentation == 2 ).sum().item() > 0:
                    # Sky
                    self.scales_and_shifts[mask_idx] = [ torch.tensor(0.).to(self.load_device), torch.tensor(0.).to(self.load_device), 1., 1e3 / BASE_SCALE ]
                else:
                    pred_depth = torch.masked_select(depth, segmentation.bool())
                    scale, shift = torch.std_mean(pred_depth)
                    depth.masked_scatter_(segmentation.bool(), (pred_depth - shift) / (scale + 1e-8))
                    self.scales_and_shifts[mask_idx] = [
                        torch.nn.Parameter(torch.tensor(0.).to(self.load_device)), 
                        torch.nn.Parameter(torch.tensor(0.).to(self.load_device)), 
                        scale.item(), 
                        shift.item() / scale.item(), 
                    ]
            self.__depth = depth[None]
        elif self.scale_and_shift_mode == 'whole':
            self.scales_and_shifts = [
                torch.nn.Parameter(torch.tensor(0.).to(self.load_device)), 
                torch.nn.Parameter(torch.tensor(0.).to(self.load_device)), 
            ]
    
    @property
    def parameters_scale(self):
        if self.scale_and_shift_mode == 'mask':
            parameters = []
            for mask_idx in self.scales_and_shifts:
                if isinstance(self.scales_and_shifts[mask_idx][0], torch.nn.Parameter):
                    parameters.append(self.scales_and_shifts[mask_idx][0])
            return parameters
        else:
            return [self.scales_and_shifts[0]]
    
    @property
    def parameters_shift(self):
        if self.scale_and_shift_mode == 'mask':
            parameters = []
            for mask_idx in self.scales_and_shifts:
                if isinstance(self.scales_and_shifts[mask_idx][1], torch.nn.Parameter):
                    parameters.append(self.scales_and_shifts[mask_idx][1])
            return parameters
        else:
            return [self.scales_and_shifts[1]]
    
    def cam_requires_grad_(self, requires_grad=False):
        self.quaternion.requires_grad_(requires_grad)
        self.T.requires_grad_(requires_grad)
    
    def scales_and_shifts_requires_grad_(self, requires_grad=False):
        if self.scale_and_shift_mode == 'mask':
            for mask_idx in self.scales_and_shifts:
                if isinstance(self.scales_and_shifts[mask_idx][0], torch.nn.Parameter):
                    self.scales_and_shifts[mask_idx][0].requires_grad_(requires_grad)
                if isinstance(self.scales_and_shifts[mask_idx][1], torch.nn.Parameter):
                    self.scales_and_shifts[mask_idx][1].requires_grad_(requires_grad)
        else:
            self.scales_and_shifts[0].requires_grad_(requires_grad)
            self.scales_and_shifts[1].requires_grad_(requires_grad)

    @property
    def projection_matrix(self):
        return getProjectionMatrix2(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1)

    @property
    def projection_matrix_inverse(self):
        return self.projection_matrix.inverse()
    
    @property
    def FoVx(self):
        return 2 * torch.atan(self.width / (2 * self.Focalx))
    
    @property
    def FoVy(self):
        return 2 * torch.atan(self.height / (2 * self.Focaly))
    
    @property
    def intrinsics(self):
        return construct_intrinsics(self.Focalx, self.Focaly, self.Offsetx, self.Offsety, self.width, self.height, sk=self.Skew).transpose(0, 1)

    @property
    def image(self):
        if self.__image is None:
            self.__image = torch.from_numpy(np.array(Image.open(self.image_path)).transpose(2, 0, 1).astype(np.float32) / 255.).to(self.load_device)
            self.__image = F.interpolate(self.__image[None, ...], (self.image_height, self.image_width), mode='bilinear', align_corners=False, antialias=True)[0]
        return self.__image
    
    @property
    def seg_mask(self):
        if self.__seg_mask is None:
            self.__seg_mask = torch.from_numpy(np.load(self.seg_mask_path)).to(self.load_device)
            self.__seg_mask = F.interpolate(self.__seg_mask[None, ...].float(), (self.image_height, self.image_width), mode='nearest')[0]
        return self.__seg_mask
    
    def init_(self, cam):
        self.quaternion.data.copy_(cam.quaternion.data)
        self.T.data.copy_(cam.T.data)
        return self

    @staticmethod
    def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    @staticmethod
    def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
        
        def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
            ret = torch.zeros_like(x)
            positive_mask = x > 0
            ret[positive_mask] = torch.sqrt(x[positive_mask])
            return ret

        batch_dim = matrix.shape[:-2]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            matrix.reshape(batch_dim + (9,)), dim=-1
        )

        q_abs = _sqrt_positive_part(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            )
        )

        quat_by_rijk = torch.stack(
            [
                torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
                torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
                torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
                torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
            ],
            dim=-2,
        )

        flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

        return quat_candidates[
            F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
        ].reshape(batch_dim + (4,))

    @property
    def R(self) -> torch.Tensor:
        return self.quaternion_to_matrix(self.quaternion)
    
    @property
    def ref_R(self) -> torch.Tensor:
        return self.quaternion_to_matrix(self.ref_quaternion)
    
    @property
    def world_view_transform(self) -> torch.Tensor:
        matrix = torch.eye(4, device=self.quaternion.device, dtype=self.quaternion.dtype, requires_grad=False)
        matrix[:3, :3] = self.R.T
        matrix[:3, 3] = self.T
        return matrix.T
    
    @property
    def ref_world_view_transform(self) -> torch.Tensor:
        matrix = torch.eye(4, device=self.quaternion.device, dtype=self.quaternion.dtype, requires_grad=False)
        matrix[:3, :3] = self.ref_R.T
        matrix[:3, 3] = self.ref_T
        return matrix.T
    
    @property
    def view_world_transform(self) -> torch.Tensor:
        return self.world_view_transform.inverse()
    
    @property
    def full_proj_transform(self) -> torch.Tensor:
        return self.world_view_transform @ self.projection_matrix
    
    @property
    def full_proj_transform_inverse(self) -> torch.Tensor:
        return self.full_proj_transform.inverse()
    
    @property
    def camera_center(self) -> torch.Tensor:
        return -self.T.view(1, 3) @ self.R.T
    
    def __repr__(self):
        return f"[Camera {self.uid}] Quaternion: {self.quaternion.detach().squeeze().cpu().numpy().tolist()}, Translation: {self.T.detach().squeeze().cpu().numpy().tolist()}"