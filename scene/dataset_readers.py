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
import sys
from dnnlib import EasyDict
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary
from scene.colmap_loader import Image as ExtrinsicsInfo
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
from pathlib import Path
from dnnlib import EasyDict

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: float
    FovX: float
    OffsetY: float
    OffsetX: float
    FocalY: float
    FocalX: float
    image: np.array
    image_path: str
    depth_path: str
    seg_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    w_gt: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readCameras(cam_extrinsics, cam_intrinsics, images_folder, depths_folder, segs_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FocalY = focal_length_x
            FocalX = focal_length_x
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            offset_x = intr.params[1]
            offset_y = intr.params[2]
            OffsetX = offset_x
            OffsetY = offset_y
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FocalY = focal_length_y
            FocalX = focal_length_x
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            offset_x = intr.params[2]
            offset_y = intr.params[3]
            OffsetX = offset_x
            OffsetY = offset_y
        else:
            assert False, f"Camera model {intr.model} not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        uid = image_name
        depth_path = os.path.join(depths_folder, image_name + ".npy")
        seg_path = os.path.join(segs_folder, image_name + '.npy')
        if os.path.exists(image_path): # and os.path.exists(depth_path) and os.path.exists(seg_path)
            image = Image.open(image_path)
            
            cam_info = CameraInfo(uid=uid, R=R, T=T, FocalY=FocalY, FocalX=FocalX, FovY=FovY, FovX=FovX, OffsetY=OffsetY, OffsetX=OffsetX, image=EasyDict(size=image.size), image_path=image_path, depth_path=depth_path, seg_path=seg_path, image_name=image_name, width=width, height=height)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readSceneInfo(path, eval, llffhold=8, num_images=-1):
    images_dir = os.path.join(path, "images")
    depths_dir = os.path.join(path, "depths")
    segs_dir = os.path.join(path, "segs")
    assert os.path.exists(images_dir), f'Cannot find the directory {images_dir} to images.'
    assert os.path.exists(depths_dir), f'Cannot find the directory {depths_dir} to depths.'
    assert os.path.exists(segs_dir), f'Cannot find the directory {segs_dir} to semantic masks.'

    # Expect intrinsics to be available
    # TODO: Estimate and optimize intrinsics
    cameras_intrinsic_file = Path(os.path.join(path, "sparse/0"))
    cam_intrinsics = None
    if os.path.exists(cameras_intrinsic_file / "cameras.bin"):
        cameras_intrinsic_file /= "cameras.bin"
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    elif os.path.exists(cameras_intrinsic_file / "cameras.txt"):
        cameras_intrinsic_file /= "cameras.txt"
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    else:
        raise NotImplementedError(f"Cannot find intrinsics file in the folder {cameras_intrinsic_file}")
    assert len(cam_intrinsics) == 1, "Currently only support 1 camera."
    
    cameras_extrinsic_file = Path(os.path.join(path, "sparse/0"))
    cam_extrinsics = None
    w_gt = True
    # If extrinsics are provided, they are used for evaluation
    if os.path.exists(cameras_extrinsic_file / "images.bin"):
        cameras_extrinsic_file /= "images.bin"
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    elif os.path.exists(cameras_extrinsic_file / "images.txt"):
        cameras_extrinsic_file /= "images.txt"
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    else:
        w_gt = False
        # Construct dummy extrinsics information
        input_image_fn_s = sorted(list(filter(lambda fn: os.path.splitext(fn)[1] in ['.png', '.jpg'], os.listdir(images_dir))))
        assert len(input_image_fn_s) > 1, f"Detect <= 1 images in the reading folder {images_dir}"
        cam_extrinsics = {}
        for image_id in range(len(input_image_fn_s)):
            cam_extrinsics[image_id] = ExtrinsicsInfo(
                id=image_id, qvec=np.array([1., 0., 0., 0.]), tvec=np.array([0., 0., 0.]), 
                camera_id=1, name=input_image_fn_s[image_id], xys=None, point3D_ids=None
            )
    
    cam_infos_unsorted = readCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=images_dir, depths_folder=depths_dir, segs_folder=segs_dir)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    print("Num of Images:", num_images)

    if eval and num_images <= 0:
        # Follow llffhold pattern to split train/test
        # But modify to swap train/test splits
        # We use this split pattern in certain scenes.
        # But metric-wise, using `num_images` should 
        # produce similar metrics.
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    elif eval and num_images > 0:
        # Use num_images to specify train/test split
        from math import floor
        length = len(cam_infos)
        interval = floor((length - num_images) / (num_images - 1))
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % (interval + 1) == 0]
        train_cam_infos[-1] = cam_infos[-1] # Ensure last frame is covered
        assert len(train_cam_infos) == num_images
        train_cam_image_name_s = {c.image_name: 1 for c in train_cam_infos}
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if not (c.image_name in train_cam_image_name_s)]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization, 
        w_gt=w_gt
    )
    return scene_info