#! /bin/bash

cd submodules/diff-gaussian-rasterization
python setup.py install
cd ../simple-knn
python setup.py install
cd ../diff-gaussian-rasterization-appr-surface
python setup.py install
cd ../QuadTreeAttention/QuadTreeAttention
python setup.py install
cd ../../fcclip/fcclip/modeling/pixel_decoder/ops
sh make.sh