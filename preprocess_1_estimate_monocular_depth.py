import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from submodules.Marigold import MariGold

device = 'cuda'

BASE_SCALING = 150
BASE_SHIFT = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", type=str, required=True)
    parser.add_argument("--depth_predictor", "-d", type=str, default='MariGold')
    parser.add_argument("--scale_factor", type=float, default=1.)
    parser.add_argument("--override", action='store_true')
    args = parser.parse_args()
    assert os.path.exists(args.source)

    depth_predictor = {'MariGold': MariGold}[args.depth_predictor]('cuda').eval().to(device).requires_grad_(False)

    image_dir = os.path.join(args.source, 'images')
    depth_dir = os.path.join(args.source, 'depths')
    os.makedirs(depth_dir, exist_ok=True)

    fn_s = sorted(os.listdir(image_dir))

    for fn in tqdm(fn_s):
        name = os.path.splitext(fn)[0]
        if os.path.exists(os.path.join(depth_dir, name + '.npy')) and not args.override:
            continue
        image = torch.from_numpy(np.array(Image.open(os.path.join(image_dir, fn)))).float().permute(2, 0, 1).to(device) / 255.
        image = torch.nn.functional.interpolate(image[None, ...], scale_factor=args.scale_factor, mode='bicubic', align_corners=False)
        depth = depth_predictor(image, inv=False).squeeze(0).cpu().numpy() * BASE_SCALING + BASE_SHIFT # Base Scaling + Shifting
        with open(os.path.join(depth_dir, name + '.npy'), 'wb') as f:
            np.save(f, depth)