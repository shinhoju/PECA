#!/usr/bin/env bash

CONFIG="/home/oem/projects/XYFormer/XY-Restormer/Motion_Deblurring/Options/Deblurring_Restormer_xy.yml"

torchrun --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt $CONFIG --launcher pytorch