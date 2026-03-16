## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer
from basicsr.models.archs.xy_restormer_arch import XYRestormer
# from basicsr.metrics.psnr_ssim import calculate_psnr
from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

parser.add_argument('--input_dir', default='./DB/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results (csv)')
parser.add_argument('--weights', default='./Motion_Deblurring/pretrained_models/motion_deblurring.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='benchmark', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
parser.add_argument('--savename', default='result6', type=str)
parser.add_argument('--start', default=103, type=int)
parser.add_argument('--end', default=205, type=int)

args = parser.parse_args()

####### Load yaml #######
yaml_file = './Motion_Deblurring/Options/Deblurring_Restormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = Restormer(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


factor = 8
dataset = args.dataset
result_dir  = os.path.join(args.result_dir, dataset)
os.makedirs(result_dir, exist_ok=True)
save_path = os.path.join(result_dir, f'{args.savename}.csv')

# sc_dirs = [os.path.join(args.input_dir, d, d) for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d, d))]
sc_dirs = [os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
sc_dirs = sorted(sc_dirs)[args.start:args.end]
for sc_dir in sc_dirs:
    print(f"Test {os.path.basename(sc_dir)} ...")
    inp_dir = os.path.join(sc_dir, 'input')
    tar_dir = os.path.join(sc_dir, 'target')
    inp_files = natsorted(glob(os.path.join(inp_dir, '*.png')))
    tar_files = natsorted(glob(os.path.join(tar_dir, '*.png')))
    metrics = dict()
    with torch.no_grad():
        for inp_file, tar_file in tqdm(zip(inp_files, tar_files), total=len(inp_files)):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(inp_file))/255.
            img = torch.from_numpy(img).permute(2,0,1)
            input_ = img.unsqueeze(0).cuda()
            
            tar = np.float32(utils.load_img(tar_file))/255.
            tar = torch.from_numpy(tar).permute(2,0,1)
            target = tar.unsqueeze(0)

            # Padding in case images are not multiples of 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

            restored_img = model_restoration(input_)
            metrics[os.path.splitext(os.path.basename(inp_file))[0]] = utils.calculate_psnr(restored_img, target, 0)
            del restored_img
            
    utils.append_scene_dict(os.path.basename(sc_dir), metrics, save_path)
