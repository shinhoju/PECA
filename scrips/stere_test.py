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
from basicsr.models.archs.stereo_xy_arch import StereoXYDeblur
from basicsr.models.archs.stereo_nafnet_arch import StereoNAFNet
from basicsr.models.archs.stereo_restormer_arch import StereoRestormer
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

parser.add_argument('--input_dir', default='/home/oem/eccv/benchmark_720/test', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/FINAL_PECA_5', type=str, help='Directory for results')
parser.add_argument('--weights', default='./experiments/FINAL_Restormer_tiny_of5_temp0.01/models/net_g_400000.pth', type=str, help='Path to weights')

args = parser.parse_args()

####### Load yaml #######
yaml_file = './Motion_Deblurring/Options/PECA_XYDeblur.yml'
# yaml_file = './Motion_Deblurring/Options/PECA_NAFNet_w64.yml'
# yaml_file = './Motion_Deblurring/Options/PECA_Restormer.yml'

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = StereoXYDeblur(**x['network_g'])
model_restoration = StereoNAFNet(**x['network_g'])
model_restoration = StereoRestormer(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration.eval()


factor = 8
result_dir  = args.result_dir
os.makedirs(result_dir, exist_ok=True)

inp_dir = os.path.join(args.input_dir, 'input')
tar_dir = os.path.join(args.input_dir, 'target')
gui_dir = os.path.join(args.input_dir, 'guide')
inp_files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
tar_files = natsorted(glob(os.path.join(tar_dir, '*.png')) + glob(os.path.join(tar_dir, '*.jpg')))
gui_files = natsorted(glob(os.path.join(gui_dir, '*.png')) + glob(os.path.join(gui_dir, '*.jpg')))

metrics_psnr = []
metrics_ssim = []
metrics = {}

with torch.no_grad():
    for inp_file, tar_file, gui_file in tqdm(zip(inp_files, tar_files, gui_files), total=len(tar_files)):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        base_name = os.path.basename(inp_file)

        img = np.float32(utils.load_img(inp_file))/255.
        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()

        gui = np.float32(utils.load_img(gui_file))/255.
        gui = torch.from_numpy(gui).permute(2,0,1)
        guide_ = gui.unsqueeze(0).cuda()
        
        tar = np.float32(utils.load_img(tar_file))/255.
        tar = torch.from_numpy(tar).permute(2,0,1)
        target = tar.unsqueeze(0)

        # Padding in case images are not multiples of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
        guide_ = F.pad(guide_, (0,padw,0,padh), 'reflect')

        restored_img = model_restoration(input_, guide_)
        restored_img = torch.clamp(restored_img,0,1)
        
        # Calculate PSNR
        psnr = calculate_psnr(restored_img, target, 0, 'CHW')
        ssim = calculate_ssim(restored_img, target, 0)
        
        metrics[base_name] = {'psnr': psnr, 'ssim': ssim}
        metrics_psnr.append(psnr)
        metrics_ssim.append(ssim)

        # Unpad images to original dimensions
        restored_img = restored_img[:,:,:h,:w]
        restored_img = torch.clamp(restored_img,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        
        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(inp_file)[-1])[0]+'_result.png')), img_as_ubyte(restored_img))

print("PSNR: ", np.array(metrics_psnr).mean())
print("SSIM: ", np.array(metrics_ssim).mean())
