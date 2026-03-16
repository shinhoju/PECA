## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

##### Data preparation file for training Restormer on the GoPro Dataset ########

import cv2
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from pdb import set_trace as stx
from joblib import Parallel, delayed
import multiprocessing
from PIL import Image
import shutil

from torchvision.transforms import Resize, InterpolationMode


def composite_blur(images):
    result = np.zeros_like(images[0], dtype=float)
    for img in images:
        result += img.astype(float)
    result /= len(images)
    return result


def get_index_from_name(fname: str) -> int:
    base = os.path.basename(fname)
    num_str = base.split('_')[0]  # '00001_l.png' -> '00001'
    return int(num_str)


def build_db(scene_dir):
    print(f"Processing {scene_dir}...")
    
    files = os.listdir(scene_dir)
    left_frs = [
        os.path.join(scene_dir, f)
        for f in files if f.lower().endswith('_l.png')
    ]
    
    right_frs = [
        os.path.join(scene_dir, f)
        for f in files if f.lower().endswith('_r.png')
    ]
    
    assert len(left_frs) == len(right_frs)
    assert len(left_frs) != 0
    
    left_frs = sorted(left_frs, key=get_index_from_name)
    right_frs = sorted(right_frs, key=get_index_from_name)
    
    num_frs = len(left_frs)
    num_pairs = num_frs - (window_size - 1)
    
    scene_name = os.path.basename(scene_dir)
    tar_dir = os.path.join(tar, scene_name)
    lr_dir = os.path.join(tar_dir, 'input')
    hr_dir = os.path.join(tar_dir, 'target')
    gd_dir = os.path.join(tar_dir, 'guide')
    
    os.makedirs(lr_dir, exist_ok=True)
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(gd_dir, exist_ok=True)
    
    name_idx = 1
    
    for pair_id in range(num_pairs):
        center_idx = pair_id + half_size
        s_idx = center_idx - half_size
        e_idx = center_idx + half_size + 1
        
        fr_paths = left_frs[s_idx:e_idx]
        
        frs = []        
        
        for p in fr_paths:
            fr = Image.open(p).convert('RGB')
            fr_np = np.array(fr, dtype=np.float32)
            frs.append(fr_np)
        avg_fr = composite_blur(frs)
        avg_fr = np.clip(avg_fr, 0, 255).astype(np.uint8)
        
        inp_img = Image.fromarray(avg_fr)
        hr_img = left_frs[center_idx]
        gd_img = right_frs[center_idx]
        
        # hr_img = Image.open(left_frs[center_idx]).convert('RGB')
        # gd_img = Image.open(right_frs[center_idx]).convert('RGB')
        
        # Resize
        # resize = Resize((720, 1280), interpolation=InterpolationMode.BICUBIC)
        # inp_img = resize(inp_img)
        # hr_img = resize(hr_img)
        # gd_img = resize(gd_img)
        
        lr_tar = os.path.join(lr_dir, f"{name_idx:05d}_blurry.png")
        hr_tar = os.path.join(hr_dir, f"{name_idx:05d}_gt.png")
        gd_tar = os.path.join(gd_dir, f"{name_idx:05d}_guide.png")
        
        inp_img.save(lr_tar)
        # hr_img.save(hr_tar)
        # gd_img.save(gd_tar)
        shutil.copy(hr_img, hr_tar)
        shutil.copy(gd_img, gd_tar)
        
        name_idx += 1
            


############ Prepare Training data ####################
num_cores = 10
window_size = 11
half_size = window_size // 2

src = './raw/'
tar = './DB/'

done_list = os.listdir(tar)

scene_dirs = [
    os.path.join(src, d, d)
    for d in os.listdir(src)
    if os.path.isdir(os.path.join(src, d, d)) and not d in done_list
    ]


Parallel(n_jobs=num_cores)(delayed(build_db)(scene_dir) for scene_dir in tqdm(scene_dirs))
