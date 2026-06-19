
# PECA: Physically-and Epipolar-constrained Cross Attention

[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-blue)](https://eccv.ecva.net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Dataset License: CC BY 4.0](https://img.shields.io/badge/Dataset%20License-CC--BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

The official pytorch implementation of the paper **[A Benchmark for Heterogeneous Stereo Deblurring with Physically- and Epipolar-constrained Cross Attention (ECCV2026)](https://openreview.net/forum?id=Wa0C2CXiEd&referrer=%5Bthe%20profile%20of%20Hoju%20Shin%5D(%2Fprofile%3Fid%3D~Hoju_Shin1))**

<p align="center">
  <img src="./assets/fig_overview.png" width="90%">
</p>

#### 저자
>Modern stereo-capable smartphones enable immersive XR content capture. However, hardware heterogeneity across camera modules often causes severe asymmetric blur artifacts. Existing methods and benchmarks largely assume homogeneous stereo setups and therefore do not explicitly address such asymmetric degradation. To bridge this gap, we present a dedicated framework for heterogeneous stereo deblurring. First, we introduce the heterogeneous stereo deblurring (HSD) dataset, constructed from real smartphone stereo captures via multi-frame integration. Second, we propose physically- and epipolar-constrained cross attention (PECA), a lightweight module that restricts cross-view matching to an epipolar search window bounded by a optics-derived disparity upper bound. By enforcing physically valid disparity constraints, PECA enables efficient and reliable cross-view feature fusion. Moreover, our confidence-weighted attention with residual fusion emphasizes cross-guided deblurring when correspondences are reliable, while naturally falling back to self-deblurring in occluded or unreliable regions. PECA is architecture-agnostic and consistently improves CNN-, Transformer-, and NAFNet-based baselines. Extensive experiments on HSD show that PECA-enhanced models achieve improved restoration performance with favorable efficiency.

## Download Dataset
HSD dataset is available on [Hugging Face](https://huggingface.co/datasets/hj-shin/HSD), [Google Drive](temp)

#### Dataset structure
```text
HSD/
├── train/
│   ├── input/
│   │   ├── 20260204_090602000_iOS_00008.png
│   │   ├── 20260204_090602000_iOS_00031.png
│   │   └── ...
│   ├── target/
│   │   ├── 20260204_090602000_iOS_00008.png
│   │   ├── 20260204_090602000_iOS_00031.png
│   │   └── ...
│   └── guide/
│       ├── 20260204_090602000_iOS_00008.png
│       ├── 20260204_090602000_iOS_00031.png
│       └── ...
│
└── test/
    ├── input/
    │   ├── 20260204_090602000_iOS_00008.png
    │   ├── 20260204_090602000_iOS_00031.png
    │   └── ...
    ├── target/
    │   ├── 20260204_090602000_iOS_00008.png
    │   ├── 20260204_090602000_iOS_00031.png
    │   └── ...
    └── guide/
        ├── 20260204_090602000_iOS_00008.png
        ├── 20260204_090602000_iOS_00031.png
        └── ...
````


## Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks and [NAFNet](https://github.com/megvii-research/NAFNet), [Restormer](https://github.com/swz30/Restormer).

```python
python 3.10.19
pytorch 2.7.0
cuda 12.8
```

```
git clone https://github.com/pknu-v-lab/PECA.git
cd PECA
pip install -r requirements.txt
python setup.py develop
```


## Quick Start

1. Generate image patches from full-resolution training images of HSD dataset
```
cd scripts
python generate_patches.py 
```

2. To train models, run
```
./train.sh./options/PECA_XYDeblur.yml
./train.sh./options/PECA_Restormer.yml
./train.sh./options/PECA_NAFNet_w64.yml
```
**Note:** The above training script uses 2 GPUs by default. To use any other number of GPUs, modify [train.sh](PECA/train.sh) and option files.


## Results and Pre-trained Models

| Method | PSNR | SSIM | pretrained models | config |
|---|---:|---:|---|---|
| XYDeblur + **Ours** | 32.22 | 0.9620 | [gdrive]() | [link](options/PECA_XYDeblur.yml) |
| Restormer + **Ours** | 32.47 | 0.9636 | [gdrive]() | [link](options/PECA_Restormer.yml) |
| NAFNet + **Ours** | 32.92 | 0.9669 | [gdrive]() | [link](options/PECA_NAFNet_w64.yml) |
