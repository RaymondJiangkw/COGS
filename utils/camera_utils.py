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
from scene.cameras import DifferentiableCamera
from tqdm import tqdm

WARNED = False

def loadCam(args, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    cam = DifferentiableCamera(
        uid=cam_info.uid, 
        image_name=cam_info.image_name, 
        load_device=args.data_device, 
        
        width=cam_info.width, 
        height=cam_info.height, 

        image_width=resolution[0], 
        image_height=resolution[1], 

        image_path=cam_info.image_path, 
        depth_path=cam_info.depth_path, 
        seg_mask_path=cam_info.seg_path, 

        quaternionOrR=torch.eye(3, dtype=torch.float32, device=args.data_device), 
        T=torch.zeros(3, dtype=torch.float32, device=args.data_device), 

        # quaternionOrR=torch.from_numpy(cam_info.R).float(), # If this is not first frame, 
        # T=torch.from_numpy(cam_info.T).float(),             # these will be reset anyways in train.py.

        ref_quaternionOrR=torch.from_numpy(cam_info.R).float(), 
        ref_T=torch.from_numpy(cam_info.T).float(), 

        Focalx=cam_info.FocalX, 
        Focaly=cam_info.FocalY, 
        Offsetx=cam_info.OffsetX, 
        Offsety=cam_info.OffsetY, 

        scale_and_shift_mode=args.scale_and_shift_mode
    ).requires_grad_(False).to(args.data_device)
    return cam

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for c in tqdm(cam_infos):
        camera_list.append(loadCam(args, c, resolution_scale))

    return camera_list
