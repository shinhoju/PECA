#!/usr/bin/env bash

CONFIG=/home/oem/eccv/XY-Restormer/options/vanilla_concat.yml
DEVICES=2,3

CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=2 --master_port=1818 basicsr/rect_train.py -opt $CONFIG --launcher pytorch