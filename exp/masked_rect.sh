#!/usr/bin/env bash

CONFIG=./options/masked_rect.yml

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=1204 basicsr/rect_train.py -opt $CONFIG --launcher pytorch