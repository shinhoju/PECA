## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import pandas as pd
import os
import cv2
import math
import torch

from pathlib import Path

def to_numpy(img):
    if isinstance(img, torch.Tensor):
        img = torch.clamp(img, 0, 1).detach().cpu().numpy()
    return img.astype(np.float64)

def calculate_psnr(img1, img2, border=0):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = to_numpy(img1)
    img2 = to_numpy(img2)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')

    return -10 * np.log10(mse)

def append_scene_dict(scene_name, data_dict, save_path):
    save_path = Path(save_path)
    df = pd.DataFrame({
        "scene_name": scene_name,
        "data_id": list(data_dict.keys()),
        "metric": [float(v) for v in data_dict.values()],
    })
    df.to_csv(save_path, mode="a", header=not save_path.exists(), index=False)

# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    _, c, h, w = img1.shape
    if border > 0:
            img1 = img1[:, :, border:h-border, border:w-border]
            img2 = img2[:, :, border:h-border, border:w-border]

    img1 = to_numpy(img1).squeeze(0)
    img2 = to_numpy(img2).squeeze(0)
    
    if c == 1:
        return ssim(img1[0], img2[0], data_range=1.0)
    elif c == 3:
        ssims = [ssim(img1[i], img2[i], data_range=1.0) for i in range(3)]
        return float(np.mean(ssims))
    else:
        raise ValueError(f"Only C=1 or C=3 supported. Got C={c}")


def ssim(img1_hw, img2_hw, data_range=1.0):
    """
    SSIM for single-channel images in [0,1], shape (H, W).
    MATLAB-like implementation with Gaussian window 11x11 sigma=1.5 and valid crop [5:-5, 5:-5].
    """
    # constants scaled by data range
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    img1 = img1_hw.astype(np.float64)
    img2 = img2_hw.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    # 'valid' region like original: remove 5 pixels on each side after filtering
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 * img1, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 * img2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12  = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())

def load_img(filepath):

    # img0 = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    # print("imread None?", img0 is None)
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)
