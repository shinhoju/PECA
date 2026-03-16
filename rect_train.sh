#!/usr/bin/env bash

CONFIG=/home/oem/eccv/XY-Restormer/options/masked_rect.yml
DEVICES="2,3"

CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=2 --master_port=3333 basicsr/rect_train.py -opt $CONFIG --launcher pytorch