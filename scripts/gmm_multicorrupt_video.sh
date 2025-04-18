#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/workspace

config=/workspace/projects/configs/occuq/occuq_mlpv5_sn.py
weight=/workspace/work_dirs/occuq_mlpv5_sn/epoch_6.pth


# Clean Video
python tools/gmm_video.py \
$config \
$weight \
--eval bbox

# MultiCorrupt
corruptions=("snow" "fog" "motionblur" "brightness" "missingcamera")
levels=("1" "2" "3")

for corruption in "${corruptions[@]}"; do
    for level in "${levels[@]}"; do
        python tools/gmm_video.py \
        $config \
        $weight \
        --eval bbox \
        --overwrite_nuscenes_root=/workspace/multicorrupt/$corruption/$level
    done
done
