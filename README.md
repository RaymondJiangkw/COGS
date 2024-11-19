## A Construct-Optimize Approach to Sparse View Synthesis without Camera Pose <br><sub>Official PyTorch implementation of the ACM SIGGRAPH 2024 paper</sub>

![Teaser image](./docs/teaser.jpg)

**A Construct-Optimize Approach to Sparse View Synthesis without Camera Pose**<br>
Kaiwen Jiang, Yang Fu, Mukund Varma T, Yash Belhe, Xiaolong Wang, Hao Su, Ravi Ramamoorthi<br>

[**Paper**](https://arxiv.org/abs/2405.03659) | [**Project**](https://raymondjiangkw.github.io/cogs.github.io/) | [**Video**](https://www.youtube.com/watch?v=0wqQnHD1R6Q)

Abstract: *Novel view synthesis from a sparse set of input images is a challenging problem of great practical interest, especially when camera poses are absent or inaccurate. Direct optimization of camera poses and usage of estimated depths in neural radiance field algorithms usually do not produce good results because of the coupling between poses and depths, and inaccuracies in monocular depth estimation. In this paper, we leverage the recent 3D Gaussian splatting method to develop a novel construct-and-optimize method for sparse view synthesis without camera poses. Specifically, we construct a solution progressively by using monocular depth and projecting pixels back into the 3D world. During construction, we optimize the solution by detecting 2D correspondences between training views and the corresponding rendered images. We develop a unified differentiable pipeline for camera registration and adjustment of both camera poses and depths, followed by back-projection. We also introduce a novel notion of an expected surface in Gaussian splatting, which is critical to our optimization. These steps enable a coarse solution, which can then be low-pass filtered and refined using standard optimization methods. We demonstrate results on the Tanks and Temples and Static Hikes datasets with as few as three widely-spaced views, showing significantly better quality than competing methods, including those with approximate camera pose information. Moreover, our results improve with more views and outperform previous InstantNGP and Gaussian Splatting algorithms even when using half the dataset.*

## Requirements
- Please refer to [3DGS](https://github.com/graphdeco-inria/gaussian-splatting/) for requirements of hardware.
- We have done all the experiments on the Linux platform with NVIDIA 3080 GPUs.
- Dependencies: see [environment.yml](./environment.yml) for exact library dependencies. You can use the following commands with Miniconda3 to create and activate your Python environment. **NOTICE:** you need to install [xformers](https://github.com/facebookresearch/xformers) and [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md), following their guidelines, manually.
    - conda env create -f environment.yml
    - conda activate cogs
    - python -m ipykernel install --user --name=cogs
    - ./install.sh

## Getting started
- Please refer to [fcclip](https://github.com/bytedance/fc-clip) for downloading `fcclip_cocopan.pth`, and put it under `submodules/fcclip`.
- Please refer to [QuadTreeAttention](https://github.com/Tangshitao/QuadTreeAttention) for downloading `indoor.ckpt` and `outdoor.ckpt`, and put them under `submodules/QuadTreeAttention`.

### Quick Demo
We provide a quick [demo](./demo.ipynb) for you to play with.

### Dataset Preparation
- For Tanks & Temples Dataset, please refer to [NoPe-NeRF](https://github.com/ActiveVisionLab/nope-nerf/) for downloading. You will then need to run `convert.py` to estimate the intrinsics and ground-truth extrinsics for evaluation.
- For Hiking Dataset, please refer to [LocalRF](https://github.com/facebookresearch/localrf) for downloading. We truncate each scene to keep first 50 frames such that 3 training views can cover the whole scene. You will then need to run `convert.py` to estimate the intrinsics and ground-truth extrinsics for evaluation.
- For your own dataset, you need to provide the intrinsics, and notice that our model assumes all views share the same intrinsics as well. You need to prepare the data under the following structure:
```
- images/                       -- Directory containing your images
- sparse/0
    - cameras.bin/cameras.txt   -- Intrinsics information in COLMAP format.
```

Afterwards, you need to run following commands to estimate monocular depths and semantic masks.
```bash
python preprocess_1_estimate_monocular_depth.py -s <path to the dataset>
python preprocess_2_estimate_semantic_mask.py -s <path to the dataset>
```
### Training
To train a scene, after preprocessing, please use
```bash
python train.py -s <path to the dataset> --eval --num_images <number of trainig views>
```
<details>
<summary><span style="font-weight: bold;">Interface for available training options (you can find default values in the 'arguments/__init__.py'):</span></summary>

Options used for constructing a coarse solution:

| Argument | Type | Description |
|:--------:|:----:|:-----------:|
| `rotation_finetune_lr` | `float` | Learning rate for the quaternion of camera |
| `translation_finetune_lr` | `float` | Learning rate for the translation of camera |
| `scale_finetune_lr` | `float` | Learning rate for the scaling per primitive for aligning the monocular depth |
| `shift_finetune_lr` | `float` | Learning rate for the translation per primitive for aligning the monocular depth |
| `register_steps` | `int` | Number of optimization steps for registering the camera pose |
| `align_steps` | `int` | Number of optimization steps for adjusting both the camera pose and monocular depth |

Options used for refinement:

| Argument | Type | Description |
|:--------:|:----:|:-----------:|
| `iterations` | `int` | Number of iterations for optimization. If this is changed, other relevant options should also be adjusted. |
| `depth_diff_tolerance` | `int` | Threshold of difference between aligned depth and rendered depth to be considered as unobserved regions |
| `farest_percent` | `float` | Percent of retained number of points after farest point down-sampling |
| `retain_percent` | `float` | Percent of retained number of points after uniform down-sampling |
| `add_frame_interval` | `int` | Interval of training views which are back-projected after registration and adjustment |
| `scale_and_shift_mode` | `'mask'` or `'whole'` | Align the monocular depth either per primitive based on mask, or as a whole |

Other hyper-parameters should be self-explaining.

</details>

### Testing
After a scene is trained, please first use
```bash
python eval.py -m <path to the saved model> --load_iteration <load iteration>
```
to estimate the extrinsics of testing views. If ground-truth extrinsics are provided, it will calculate the metrics of estimated extrinsics of training views as well.

For the Hiking dataset, we use
```bash
python eval.py -m <path to the saved model> --load_iteration <load iteration> --rgb_only
```
to estimate the extrinsics of testing views.

After registering the testing views, please use `render.py` and `metrics.py` to evaluate the novel view synthesis performance.

### Tips
As to training, you may need to tweak the hyper-parameters to adapt to different scenes for best performance. For example, 
- If the distance between consecutive training frames is relatively large, `register_steps` and `align_steps` are recommended to increase.
- If the number of training views are relatively large (e.g., 30 or 60 frames), it is recommended to set `add_frame_interval` to be larger than 1, and decrease `register_steps` and `align_steps` to avoid alignment and back-projection for unnecessary frames and speed up the process. The number of iterations is also recommended to increase.
    - E.g., we use `add_frame_interval = 10`, `register_steps = 50`, `align_steps = 100`, number of iterations `= 30000` for the Horse scene with `60/120` training views.
- If the training frame contains too many primitives, which makes the number of correspondence points located on each primitive relatively low, setting `scale_and_shift_mode` to `whole`, and `depth_diff_tolerance` to `1e9` may produce better results.
- Our method focuses on scene-level sparse view synthesis. If you want to apply this algorithm to object-level sparse view synthesis, you may want to adjust the `BASE_SCALING` and `BASE_SHFIT` in `preprocess_1_estimate_monocular_depth.py` and `scene/cameras.py`. Besides, you may want to adjust the `depth_diff_tolerance` as well. For example, dividing their initial values by 10 should be helpful.

As to testing, we evaluate at both 3000 and 9000 iterations' checkpoints, and use the better one.

## FAQs
- **I don't have intrinsics, and fail to estimate the intrinsics as well.** Usually, the intrinsics come with your camera. You may use your camera to carefully and densely capture a scene, and use its estimated intrinsics. Meanwhile, our rasterizer supports differentiation through the intrinsic parameters. You may try to optimize the intrinsic parameters as well if you can manage to get a coarse estimation in the first place (*Caution: We haven't tested it yet*). Notice that, inheriting from the 3DGS, only perspective camera is supported, and the principal point must lie at the center, and there is no skew.
- **I want to use SIBR viewer provided by 3DGS to inspect the scene.** Since we do not export registered cameras parameters into a format that can be read by the SIBR viewer (*Maybe added later*), a work-around is to first apply COLMAP which sees all the views to estimate the ground-truth extrinsics, and then ensure the extrinsics of the first frame in our pipeline aligns with that in COLMAP by *Commenting L55-56, and Uncommenting L58-59 in `./utils/camera_utils.py`*. If you want to evaluate the pose metrics, it is also necessary to align the pose of the first frame as well.
- **I want to speed up the refinement process.** Currently, our refinement process utilizes the same rasterizer as that in the construction phase. Since we just use its differentiability of camera parameters, it is possible to remove unnecessary parts (i.e., ray-sphere intersection) in the rasterizer during the refinement to alleviate control divergence and speed up the training.

## Acknowledgement
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting/). We also utilize [FC-CLIP](https://github.com/bytedance/fc-clip), [MariGold](https://github.com/prs-eth/marigold), and [QuadTreeAttention](https://github.com/Tangshitao/QuadTreeAttention). We thank authors for their great repos.

## Citation
```
@article{COGS2024,
    title={A Construct-Optimize Approach to Sparse View Synthesis without Camera Pose},
    author={Jiang, Kaiwen and Fu, Yang and Varma T, Mukund and Belhe, Yash and Wang, Xiaolong and Su, Hao and Ramamoorthi, Ravi},
    journal={SIGGRAPH},
    year={2024}
}
```