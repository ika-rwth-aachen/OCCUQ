#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,4
export PYTHONPATH=$PYTHONPATH:/workspace

./tools/dist_train.sh \
/workspace/projects/configs/occuq/occuq_mlpv5_sn.py 4 \
/workspace/work_dirs/occuq_mlpv5_sn