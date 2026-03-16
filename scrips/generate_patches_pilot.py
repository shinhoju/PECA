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


def build_scene_pairs(src_root):
    scene_dirs = natsorted([d for d in glob(os.path.join(src_root, '*')) if os.path.isdir(d)])
    pairs = []

    for scene_dir in scene_dirs:
        scene_id = os.path.basename(scene_dir)

        sharp_dir = os.path.join(scene_dir, 'index_gt')
        blurry_dir = os.path.join(scene_dir, 'index_blurry')
        guide_dir = os.path.join(scene_dir, 'index_guide')

        if not (os.path.isdir(sharp_dir) and os.path.isdir(blurry_dir) and os.path.isdir(guide_dir)):
            continue

        sharp_files = natsorted(glob(os.path.join(sharp_dir, '*.png')))
        for hr_path in sharp_files:
            base = os.path.splitext(os.path.basename(hr_path))[0]  # e.g. "0001_sharp"
            # frame id는 앞의 "0001"만 쓰고 싶으니 split('_')[0]
            frame_id = base.split('_')[0]

            # 대응되는 blurry 파일 경로 만들기
            # e.g. "0001_blurry.png"
            ext = os.path.splitext(hr_path)[1]
            lr_path = os.path.join(blurry_dir, f"{frame_id}_blurry{ext}")
            gd_path = os.path.join(guide_dir, f"{frame_id}_guide{ext}")

            # 저장 stem: [장면번호]_[프레임ID]_sharp
            out_stem = f"{scene_id}_{frame_id}"
            pairs.append((lr_path, hr_path, gd_path, out_stem))

    return pairs

def train_files(file_):
    lr_file, hr_file, gd_file, out_stem = file_
    lr_img = cv2.imread(lr_file)
    hr_img = cv2.imread(hr_file)
    gd_img = cv2.imread(gd_file)
    
    num_patch = 0
    w, h = lr_img.shape[:2]
    
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-patch_size, patch_size-overlap, dtype=int))
        h1 = list(np.arange(0, h-patch_size, patch_size-overlap, dtype=int))
        w1.append(w-patch_size)
        h1.append(h-patch_size)
        for i in w1:
            for j in h1:
                num_patch += 1
                
                lr_patch = lr_img[i:i+patch_size, j:j+patch_size,:]
                hr_patch = hr_img[i:i+patch_size, j:j+patch_size,:]
                gd_patch = gd_img[i:i+patch_size, j:j+patch_size,:]
                
                lr_savename = os.path.join(lr_tar, out_stem + '-' + str(num_patch) + '.png')
                hr_savename = os.path.join(hr_tar, out_stem + '-' + str(num_patch) + '.png')
                gd_savename = os.path.join(gd_tar, out_stem + '-' + str(num_patch) + '.png')
                
                cv2.imwrite(lr_savename, lr_patch)
                cv2.imwrite(hr_savename, hr_patch)
                cv2.imwrite(gd_savename, gd_patch)

    else:
        lr_savename = os.path.join(lr_tar, out_stem + '.png')
        hr_savename = os.path.join(hr_tar, out_stem + '.png')
        gd_savename = os.path.join(gd_tar, out_stem + '.png')
        
        cv2.imwrite(lr_savename, lr_img)
        cv2.imwrite(hr_savename, hr_img)
        cv2.imwrite(gd_savename, gd_img)

def val_files(file_):
    lr_file, hr_file, gd_file, out_stem = file_
    lr_img = cv2.imread(lr_file)
    hr_img = cv2.imread(hr_file)
    gd_img = cv2.imread(gd_file)

    lr_savename = os.path.join(lr_tar, out_stem + '.png')
    hr_savename = os.path.join(hr_tar, out_stem + '.png')
    gd_savename = os.path.join(gd_tar, out_stem + '.png')

    w, h = lr_img.shape[:2]

    i = (w-val_patch_size)//2
    j = (h-val_patch_size)//2

    lr_patch = lr_img[i:i+val_patch_size, j:j+val_patch_size,:]
    hr_patch = hr_img[i:i+val_patch_size, j:j+val_patch_size,:]
    gd_patch = gd_img[i:i+val_patch_size, j:j+val_patch_size,:]

    cv2.imwrite(lr_savename, lr_patch)
    cv2.imwrite(hr_savename, hr_patch)
    cv2.imwrite(gd_savename, gd_patch)

############ Prepare Training data ####################
num_cores = 10
patch_size = 512
overlap = 256
p_max = 0

src = '/home/oem/data/pilot_v2/train'
tar = '/home/oem/projects-hoju/XY-Restormer/Motion_Deblurring/Datasets/train/pilot_v2'

lr_tar = os.path.join(tar, 'input_crops')
hr_tar = os.path.join(tar, 'target_crops')
gd_tar = os.path.join(tar, 'guide_crops')

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar, exist_ok=True)
os.makedirs(gd_tar, exist_ok=True)

files = build_scene_pairs(src)

Parallel(n_jobs=num_cores)(delayed(train_files)(file_) for file_ in tqdm(files))


############ Prepare validation data ####################
val_patch_size = 256
src = '/home/oem/data/pilot_v2/val'
tar = '/home/oem/projects-hoju/XY-Restormer/Motion_Deblurring/Datasets/val/pilot_v2'

lr_tar = os.path.join(tar, 'input_crops')
hr_tar = os.path.join(tar, 'target_crops')
gd_tar = os.path.join(tar, 'guide_crops')

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar, exist_ok=True)
os.makedirs(gd_tar, exist_ok=True)

files = build_scene_pairs(src)

Parallel(n_jobs=num_cores)(delayed(val_files)(file_) for file_ in tqdm(files))
