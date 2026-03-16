## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import numpy as np
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
# from basicsr.metrics.psnr_ssim import calculate_psnr
from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

parser.add_argument('--input_dir', default='./datasets/test/benchmark_720', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Restormer_vanilla', type=str, help='Directory for results')
parser.add_argument('--weights', default='./experiments/Restormer_vanilla/models/net_g_400000.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']

args = parser.parse_args()

####### Load yaml #######
# yaml_file = './options/train/Pilot/rect.yml'
yaml_file = './options/vanilla.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

# model_restoration = RectifiedNAFNetLocal(**x['network_g'])
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

inp_dir = os.path.join(args.input_dir, 'input')
tar_dir = os.path.join(args.input_dir, 'target')
inp_files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
tar_files = natsorted(glob(os.path.join(tar_dir, '*.png')) + glob(os.path.join(tar_dir, '*.jpg')))
metrics_psnr = []
metrics_ssim = []
with torch.no_grad():
    for inp_file, tar_file in tqdm(zip(inp_files, tar_files), total=len(tar_files)):
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
        
        # Calculate PSNR
        metrics_psnr.append(utils.calculate_psnr(restored_img, target, 0))
        metrics_ssim.append(utils.calculate_ssim(restored_img, target, 0))

        # Unpad images to original dimensions
        restored_img = restored_img[:,:,:h,:w]
        restored_img = torch.clamp(restored_img,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        
        # utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(inp_file)[-1])[0]+'_result.png')), img_as_ubyte(restored_img))

print("PSNR: ", np.array(metrics_psnr).mean())
print("SSIM: ", np.array(metrics_ssim).mean())
