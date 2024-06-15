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
import torch
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import readSceneInfo
from scene.gaussian_model import BindableGaussianModel, GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos

class Scene:

    gaussians : GaussianModel

    def __init__(self,
        args: ModelParams, 
        gaussians: GaussianModel, 
        load_iteration=None, 
        resolution_scales=[1.0], 
        train_cameras_override=None, 
        test_cameras_override=None, 
        cameras_extent=None, 
        init_pc: BasicPointCloud=None, 
        assert_test_cameras=False, 
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        load_path = self.model_path

        if load_iteration is not None:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(load_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        self.scene_info = readSceneInfo(args.source_path, args.eval, args.llffhold, args.num_images)
        self.cameras_extent = self.scene_info.nerf_normalization["radius"] if cameras_extent is None else cameras_extent

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale, args) if train_cameras_override is None else train_cameras_override[resolution_scale]
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale, args) if test_cameras_override is None else test_cameras_override[resolution_scale]

        if self.loaded_iter is not None:
            self.gaussians.load_ply(os.path.join(load_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            
            if os.path.exists(os.path.join(self.model_path, "camera/iteration_{}".format(self.loaded_iter), "train_cameras.pkl")):
                print("Load Training Camera Parameters ...")
                state_dicts = torch.load(os.path.join(self.model_path, "camera/iteration_{}".format(self.loaded_iter), "train_cameras.pkl"))
                for scale in self.train_cameras:
                    for cam in self.train_cameras[scale]:
                        cam.load_state_dict(state_dicts[scale][cam.image_name])
            
            if os.path.exists(os.path.join(self.model_path, "camera/iteration_{}".format(self.loaded_iter), "test_cameras.pkl")):
                print("Load Testing Camera Parameters ...")
                state_dicts = torch.load(os.path.join(self.model_path, "camera/iteration_{}".format(self.loaded_iter), "test_cameras.pkl"))
                for scale in self.test_cameras:
                    for cam in self.test_cameras[scale]:
                        cam.load_state_dict(state_dicts[scale][cam.image_name])
            elif assert_test_cameras:
                raise ValueError("Cannot find registered test cameras. Please run `eval.py` first.")
        else:
            self.gaussians.create_from_pcd(init_pc, min(self.cameras_extent, 5.))

    def save(self, iteration, skip_test=True):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        camera_path = os.path.join(self.model_path, "camera/iteration_{}".format(iteration))
        os.makedirs(camera_path, exist_ok=True)
        torch.save({scale: {cam.image_name: cam.state_dict() for cam in self.train_cameras[scale] } for scale in self.train_cameras}, os.path.join(camera_path, "train_cameras.pkl"))
        if not skip_test:
            torch.save({scale: {cam.image_name: cam.state_dict() for cam in self.test_cameras[scale] } for scale in self.test_cameras}, os.path.join(self.model_path, "camera/iteration_{}".format(self.loaded_iter), "test_cameras.pkl"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

class SceneFromScratch:
    gaussians : BindableGaussianModel

    def __init__(self, 
        args : ModelParams, 
        gaussians : BindableGaussianModel, 
        resolution_scales=[1.0], 
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        self.train_cameras = {}
        self.test_cameras = {}

        self.scene_info = readSceneInfo(args.source_path, args.eval, args.llffhold, args.num_images)
        self.cameras_extent = self.scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale, args)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
