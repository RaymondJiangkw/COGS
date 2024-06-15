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
import math
import numpy as np
from typing import NamedTuple
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

def my_triangulate_points(
    P1: torch.Tensor, P2: torch.Tensor, points1: torch.Tensor, points2: torch.Tensor
):
    # allocate and construct the equations matrix with shape (*, 4, 4)
    points_shape = max(points1.shape, points2.shape)  # this allows broadcasting
    X = torch.zeros(points_shape[:-1] + (4, 4)).type_as(points1)

    for i in range(4):
        X[..., 0, i] = points1[..., 0] * P1[..., 2:3, i] - P1[..., 0:1, i]
        X[..., 1, i] = points1[..., 1] * P1[..., 2:3, i] - P1[..., 1:2, i]
        X[..., 2, i] = points2[..., 0] * P2[..., 2:3, i] - P2[..., 0:1, i]
        X[..., 3, i] = points2[..., 1] * P2[..., 2:3, i] - P2[..., 1:2, i]
    
    A = X[..., :3]
    B = -X[..., 3]
    return torch.linalg.lstsq(A, B).solution

# Borrowed from PyTorch3D
def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling

def quaternion_raw_multiply_with_transpose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = - aw * bw - ax * bx - ay * by - az * bz
    ox = - aw * bx + ax * bw + ay * bz - az * by
    oy = - aw * by - ax * bz + ay * bw + az * bx
    oz = - aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)
def quaternion_multiply_with_transpose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply_with_transpose(a, b)
    return standardize_quaternion(ab)

def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

class SavedPointCloud(NamedTuple):
    xyz: torch.Tensor
    features_dc: torch.Tensor
    features_rest: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrix2(znear, zfar, fovX, fovY):
    tanHalfFovY = torch.tan((fovY / 2))
    tanHalfFovX = torch.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=fovX.device, dtype=fovX.dtype, requires_grad=False)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def construct_intrinsics(focal_x, focal_y, offset_x, offset_y, width, height, normalize=True, sk=0):
    if normalize:
        focal_x = focal_x / width
        focal_y = focal_y / height
        offset_x = offset_x / width
        offset_y = offset_y / height
    if not isinstance(focal_x, torch.Tensor):
        return torch.tensor([[focal_x, sk, offset_x], [0, focal_y, offset_y], [0, 0, 1]])
    intrinsics = torch.zeros((3, 3), device=focal_x.device, dtype=focal_x.dtype, requires_grad=False)
    intrinsics[0, 0] = focal_x
    intrinsics[0, 1] = sk
    intrinsics[0, 2] = offset_x
    intrinsics[1, 1] = focal_y
    intrinsics[1, 2] = offset_y
    intrinsics[2, 2] = 1.
    return intrinsics

def construct_rays_by_rel_points(cam2world_matrix, cam_rel_points, unit_dirs=True):
    ray_origins = cam2world_matrix[3, :3][None, ...]
    ray_dirs = (cam_rel_points @ cam2world_matrix)[..., :3] - ray_origins
    if unit_dirs:
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=-1)
    else:
        ray_dirs = (cam2world_matrix[:3, :3] @ ray_dirs.T).T
        ray_dirs = -ray_dirs / ray_dirs[..., 2:] # Certain Convention
        ray_dirs = (cam2world_matrix[:3, :3].T @ ray_dirs.T).T
    return ray_origins, ray_dirs

def construct_rays_by_uv(cam2world_matrix, intrinsics, uv, unit_dirs=False):
    N, M = cam2world_matrix.shape[0], uv.shape[0]
    cam_locs_world = cam2world_matrix[:, :3, 3]
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    uv = uv.unsqueeze(0)

    x_cam = uv[:, :, 0].view(N, -1)
    y_cam = uv[:, :, 1].view(N, -1)
    z_cam = -torch.ones((N, M), device=cam2world_matrix.device)

    x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y_cam/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
    y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

    cam_rel_points = torch.stack((x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1) # (B, M, 4)
    world_rel_points = torch.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3] # (B, N, 3)

    ray_dirs = world_rel_points - cam_locs_world[:, None, :]
    # print(ray_dirs)
    if unit_dirs:
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)
    else:
        ray_dirs = torch.bmm(cam2world_matrix[:, :3, :3].permute(0, 2, 1), ray_dirs.permute(0, 2, 1)).permute(0, 2, 1)
        ray_dirs = -ray_dirs / ray_dirs[..., 2:] # Certain Convention
        ray_dirs = torch.bmm(cam2world_matrix[:, :3, :3], ray_dirs.permute(0, 2, 1)).permute(0, 2, 1)
    # print(ray_dirs)
    ray_origins = cam_locs_world.unsqueeze(1)

    return ray_origins[0], ray_dirs[0]

def construct_uv(H, W, device, H_offset = 0., W_offset = 0.):
    uv = torch.stack(torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device) * (1./H) + ((0.5 + H_offset)/H), 
        torch.arange(W, dtype=torch.float32, device=device) * (1./W) + ((0.5 + W_offset)/W), 
    indexing='ij'))
    return uv.flip(0).reshape(2, -1).transpose(1, 0)

def construct_unit_dirs_by_uv(uv, intrinsics):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    sk = intrinsics[0, 1]

    x_cam = uv[:, 0]
    y_cam = uv[:, 1]
    z_cam = -torch.ones_like(x_cam)

    x_lift = (x_cam - cx + cy * sk / fy - sk * y_cam / fy) / fx * z_cam
    y_lift = (y_cam - cy) / fy * z_cam

    return torch.nn.functional.normalize(torch.stack((x_lift, y_lift, z_cam), dim=-1))

def construct_rays(cam2world_matrix, intrinsics, resolutions, unit_dirs=True):
    H, W = resolutions
    N, M = cam2world_matrix.shape[0], H * W
    cam_locs_world = cam2world_matrix[:, :3, 3]
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    uv = torch.stack(torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=cam2world_matrix.device) * (1./H) + (0.5/H), 
        torch.arange(W, dtype=torch.float32, device=cam2world_matrix.device) * (1./W) + (0.5/W), 
    indexing='ij'))
    uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
    uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

    x_cam = uv[:, :, 0].view(N, -1)
    y_cam = uv[:, :, 1].view(N, -1)
    z_cam = -torch.ones((N, M), device=cam2world_matrix.device)

    x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y_cam/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
    y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

    cam_rel_points = torch.stack((x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1) # (B, H*W, 4)
    world_rel_points = torch.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3] # (B, N, 3)

    ray_dirs = world_rel_points - cam_locs_world[:, None, :] # (B, N, 3)
    if unit_dirs:
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)
    else:
        ray_dirs = torch.bmm(cam2world_matrix[:, :3, :3].permute(0, 2, 1), ray_dirs.permute(0, 2, 1)).permute(0, 2, 1)
        ray_dirs = -ray_dirs / ray_dirs[..., 2:] # Certain Convention
        ray_dirs = torch.bmm(cam2world_matrix[:, :3, :3], ray_dirs.permute(0, 2, 1)).permute(0, 2, 1)

    ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

    return ray_origins, ray_dirs, cam_rel_points

def get_appr_ellipsoid_scaling(scales: torch.Tensor):
    return scales * 2.0

def get_appr_ellipsoid_scaling_inverse(ellipsoid_scale: torch.Tensor):
    return ellipsoid_scale / 2.0