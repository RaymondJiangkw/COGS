## A Construct-Optimize Approach to Sparse View Synthesis without Camera Pose <br><sub>Official PyTorch implementation of the ACM SIGGRAPH 2024 paper</sub>

![Teaser image](./docs/teaser.jpg)

**A Construct-Optimize Approach to Sparse View Synthesis without Camera Pose**<br>
Kaiwen Jiang, Yang Fu, Mukund Varma T, Yash Belhe, Xiaolong Wang, Hao Su, Ravi Ramamoorthi<br>

[**Paper**](https://arxiv.org/abs/2405.03659) | [**Project**](https://raymondjiangkw.github.io/cogs.github.io/) | [**Video**](https://www.youtube.com/watch?v=0wqQnHD1R6Q)

Abstract: *Novel view synthesis from a sparse set of input images is a challenging problem of great practical interest, especially when camera poses are absent or inaccurate. Direct optimization of camera poses and usage of estimated depths in neural radiance field algorithms usually do not produce good results because of the coupling between poses and depths, and inaccuracies in monocular depth estimation. In this paper, we leverage the recent 3D Gaussian splatting method to develop a novel construct-and-optimize method for sparse view synthesis without camera poses. Specifically, we construct a solution progressively by using monocular depth and projecting pixels back into the 3D world. During construction, we optimize the solution by detecting 2D correspondences between training views and the corresponding rendered images. We develop a unified differentiable pipeline for camera registration and adjustment of both camera poses and depths, followed by back-projection. We also introduce a novel notion of an expected surface in Gaussian splatting, which is critical to our optimization. These steps enable a coarse solution, which can then be low-pass filtered and refined using standard optimization methods. We demonstrate results on the Tanks and Temples and Static Hikes datasets with as few as three widely-spaced views, showing significantly better quality than competing methods, including those with approximate camera pose information. Moreover, our results improve with more views and outperform previous InstantNGP and Gaussian Splatting algorithms even when using half the dataset.*

**Coming soon!**

## Citation
```
@article{COGS2024,
    title={A Construct-Optimize Approach to Sparse View Synthesis without Camera Pose},
    author={Jiang, Kaiwen and Fu, Yang and Varma T, Mukund and Belhe, Yash and Wang, Xiaolong and Su, Hao and Ramamoorthi, Ravi},
    journal={SIGGRAPH},
    year={2024}
}
```