#!/usr/bin/env bash

CONFIG="./XY-Restormer/Motion_Deblurring/Options/rect_temp_sep.yml"
DEVICES="0,1"

SEARCH_RANGES=(5 6 7 8)
TEMPERATURE=0.01
ENC_MODE="sep"
LIN=false
SEED=100

BASE_EXP_NAME="RR"

for RANGE in "${SEARCH_RANGES[@]}"
do
    echo "=================================================="
    echo "   Starting Experiment: search_range = $RANGE"
    echo "=================================================="

    TMP_CONFIG="$(mktemp)"
    cp "$CONFIG" "$TMP_CONFIG"

    yq -i ".network_g.search_range = $RANGE" "$TMP_CONFIG"
    yq -i ".network_g.temperature = $TEMPERATURE" "$TMP_CONFIG"
    yq -i ".network_g.enc_mode = $ENC_MODE" "$TMP_CONFIG"
    yq -i ".network_g.lin = $LIN" "$TMP_CONFIG"
    yq -i ".manual_seed = $SEED" "$TMP_CONFIG"

    EXP_NAME="${BASE_EXP_NAME}_of${RANGE}"
    yq -i ".name = \"$EXP_NAME\"" "$TMP_CONFIG"

    CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=2 --master_port=4322 basicsr/rect_train.py -opt $TMP_CONFIG --launcher pytorch

    rm "$TMP_CONFIG"
    echo "Finished search_range = $RANGE"
    echo ""
    sleep 5
done
