#!/usr/bin/env bash

CONFIG=./options/XYDeblur.yml
DEVICES="0"

torchrun --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt $CONFIG --launcher pytorch