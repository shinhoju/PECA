#!/usr/bin/env bash

CONFIG=./options/masked_rect.yml
DEVICES="0,1"

CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=2 --master_port=3333 basicsr/stereo_train.py -opt $CONFIG --launcher pytorch