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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    tanfovx, 
    tanfovy, 
    intrinsicmatrix, 
    viewmatrix, 
    worldmatrix, 
    projmatrix, 
    campos, 
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        tanfovx, 
        tanfovy, 
        intrinsicmatrix, 
        viewmatrix, 
        worldmatrix, 
        projmatrix, 
        campos, 
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tanfovx, 
        tanfovy, 
        intrinsicmatrix, 
        viewmatrix, 
        worldmatrix, 
        projmatrix, 
        campos, 
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            intrinsicmatrix, 
            viewmatrix,
            worldmatrix, 
            projmatrix,
            tanfovx,
            tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, xy, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, xy, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(tanfovx, tanfovy, intrinsicmatrix, viewmatrix, worldmatrix, projmatrix, campos, colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, depth, xy)
        return color, radii, depth, xy

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth, grad_xy):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        tanfovx, tanfovy, intrinsicmatrix, viewmatrix, worldmatrix, projmatrix, campos, colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, out_depth, out_xy = ctx.saved_tensors
        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                intrinsicmatrix, 
                viewmatrix, 
                worldmatrix, 
                projmatrix, 
                tanfovx, 
                tanfovy, 
                out_depth, 
                out_xy, 
                grad_out_color,
                grad_depth,
                grad_xy, 
                sh, 
                raster_settings.sh_degree, 
                campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_tanfovx, grad_tanfovy, grad_intrinsicmatrix, grad_viewmatrix, grad_worldmatrix, grad_projmatrix, grad_campos, grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_tanfovx, grad_tanfovy, grad_intrinsicmatrix, grad_viewmatrix, grad_worldmatrix, grad_projmatrix, grad_campos, grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
        # print('grad_viewmatrix', grad_viewmatrix)
        # print('grad_worldmatrix', grad_worldmatrix)
        # print('grad_projmatrix', grad_projmatrix)
        # print('grad_campos', grad_campos)
        if torch.isnan(grad_tanfovx).any() or \
            torch.isnan(grad_tanfovy).any() or \
            torch.isnan(grad_intrinsicmatrix).any() or \
            torch.isnan(grad_viewmatrix).any() or \
            torch.isnan(grad_worldmatrix).any() or \
            torch.isnan(grad_projmatrix).any() or \
            torch.isnan(grad_campos).any() or \
            torch.isnan(grad_means2D).any() or \
            torch.isnan(grad_opacities).any() or \
            torch.isnan(grad_means3D).any() or \
            torch.isnan(grad_sh).any() or \
            torch.isnan(grad_scales).any() or \
            torch.isnan(grad_rotations).any():
                print('grad_tanfovx', grad_tanfovx)
                print('grad_tanfovy', grad_tanfovy)
                print('grad_intrinsicmatrix', grad_intrinsicmatrix)
                print('grad_viewmatrix', grad_viewmatrix)
                print('grad_worldmatrix', grad_worldmatrix)
                print('grad_projmatrix', grad_projmatrix)
                print('grad_campos', grad_campos)
                print('grad_means2D', grad_means2D)
                print('grad_opacities', grad_opacities)
                print('grad_means3D', grad_means3D)
                print('grad_sh', grad_sh)
                print('grad_scales', grad_scales)
                print('grad_rotations', grad_rotations)
                torch.save(cpu_deep_copy_tuple(args), "snapshot_bw.dump")
                grad_tanfovx = torch.zeros_like(grad_tanfovx)
                grad_tanfovy = torch.zeros_like(grad_tanfovy)
                grad_intrinsicmatrix = torch.zeros_like(grad_intrinsicmatrix)
                grad_viewmatrix = torch.zeros_like(grad_viewmatrix)
                grad_worldmatrix = torch.zeros_like(grad_worldmatrix)
                grad_projmatrix = torch.zeros_like(grad_projmatrix)
                grad_campos = torch.zeros_like(grad_campos)
                grad_means2D = torch.zeros_like(grad_means2D)
                grad_opacities = torch.zeros_like(grad_opacities)
                grad_means3D = torch.zeros_like(grad_means3D)
                grad_sh = torch.zeros_like(grad_sh)
                grad_scales = torch.zeros_like(grad_scales)
                grad_rotations = torch.zeros_like(grad_rotations)
                raise ValueError("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
        grads = (
            grad_tanfovx.reshape_as(tanfovx), 
            grad_tanfovy.reshape_as(tanfovy), 
            grad_intrinsicmatrix.reshape_as(intrinsicmatrix), 
            grad_viewmatrix.reshape_as(viewmatrix), 
            grad_worldmatrix.reshape_as(worldmatrix), 
            grad_projmatrix.reshape_as(projmatrix), 
            grad_campos.reshape_as(campos), 
            grad_means3D,
            grad_means2D,
            grad_sh.reshape_as(sh),
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    # tanfovx : float
    # tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    # viewmatrix : torch.Tensor
    # projmatrix : torch.Tensor
    sh_degree : int
    # campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    @torch.no_grad()
    def markVisible(self, positions, viewmatrix, projmatrix):
        visible = _C.mark_visible(
            positions,
            viewmatrix,
            projmatrix
        )
        return visible

    def forward(self, tanfovx, tanfovy, intrinsicmatrix, viewmatrix, worldmatrix, projmatrix, campos, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            tanfovx, 
            tanfovy, 
            intrinsicmatrix, 
            viewmatrix, 
            worldmatrix, 
            projmatrix, 
            campos, 
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

