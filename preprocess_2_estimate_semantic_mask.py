import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from submodules.fcclip import MaskPredictor

import numpy as np
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

device = 'cuda'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", type=str, required=True)
    parser.add_argument("--mask_predictor", "-m", type=str, default='fcclip')
    parser.add_argument("--override", action='store_true')
    parser.add_argument("--scale_factor", type=float, default=1.)
    
    args = parser.parse_args()
    assert os.path.exists(args.source)

    if args.mask_predictor == 'fcclip':
        mask_generator = MaskPredictor(0.9)

    image_dir = os.path.join(args.source, 'images')
    segmentation_dir = os.path.join(args.source, 'segs')
    os.makedirs(segmentation_dir, exist_ok=True)

    fn_s = sorted(os.listdir(image_dir))

    for fn in tqdm(fn_s):
        name = os.path.splitext(fn)[0]
        if os.path.exists(os.path.join(segmentation_dir, name + '.npy')) and not args.override:
            continue
        segmentations = []
        image = torch.from_numpy(np.array(Image.open(os.path.join(image_dir, fn)))).float().permute(2, 0, 1).to(device) / 255.
        image = torch.nn.functional.interpolate(image[None, ...], scale_factor=args.scale_factor, mode='bilinear', align_corners=False, antialias=True)
        image = render_tensor(image)
        masks = mask_generator(image)
        mask = masks[0]['panoptic_seg'][0]
        info = masks[0]['panoptic_seg'][1]
        categories = torch.unique(mask).cpu().numpy().tolist()
        for idx in categories:
            i = list(filter(lambda x: x['id'] == idx, info))
            multiplier = 1.
            if len(i) > 0 and i[0]['category_id'] in [1422]:
                multiplier = 2. # Standing for very distant area
            segmentations.append((multiplier * (mask == idx).float()).cpu().numpy())
        segmentations = np.stack(segmentations, axis=0)
        with open(os.path.join(segmentation_dir, name + '.npy'), 'wb') as f:
            np.save(f, segmentations)