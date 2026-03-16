#!/usr/bin/env bash

CONFIG="./XY-Restormer/Motion_Deblurring/Options/rect_temp_sep.yml"
DEVICES="0,1"

TMP_CONFIG="$(mktemp)"
trap 'rm -f "$TMP_CONFIG"' EXIT
cp "$CONFIG" "$TMP_CONFIG"

yq -i '.network_g.search_range = 5' "$TMP_CONFIG"
yq -i '.network_g.temperature = 0.01' "$TMP_CONFIG"
yq -i '.network_g.enc_mode = "sep"' "$TMP_CONFIG"
yq -i '.network_g.lin = false' "$TMP_CONFIG"


CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=2 --master_port=4322 basicsr/rect_train.py -opt $TMP_CONFIG --launcher pytorch