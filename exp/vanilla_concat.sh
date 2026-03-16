#!/usr/bin/env bash

CONFIG=./options/vanilla_concat.yml
DEVICES=2,3

CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=2 --master_port=4445 basicsr/rect_train.py -opt $CONFIG --launcher pytorch