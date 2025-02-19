# Train and Test

Check folder `/workspace/scripts` for scripts to train and test the models.


Train OCCUQ with 4 A100 GPUs
```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,4
export PYTHONPATH=$PYTHONPATH:/workspace

./tools/dist_train.sh \
/workspace/projects/configs/occuq/occuq_mlpv5_sn.py 4 \
/workspace/work_dirs/occuq_mlpv5_sn
```

Eval OCCUQ with 1 A100 GPU
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/workspace

config=/workspace/projects/configs/occuq/occuq_mlpv5_sn.py 
weight=/workspace/work_dirs/occuq_mlpv5_sn/epoch_6.pth

python tools/gmm_evaluate.py \
$config \
$weight \
--eval bbox
```

For video generation, run the following command:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/workspace

config=/workspace/projects/configs/occuq/occuq_mlpv5_sn.py 
weight=/workspace/work_dirs/occuq_mlpv5_sn/epoch_6.pth

python tools/gmm_video.py \
$config \
$weight \
--eval bbox
```

To perform evaluation on MultiCorrupt run the following command:
```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/workspace

config=/workspace/projects/configs/occuq/occuq_mlpv5_sn.py
weight=/workspace/work_dirs/occuq_mlpv5_sn/epoch_6.pth

python tools/gmm_fit.py \
$config \
$weight \
--eval bbox

# Clean Evaluation
python tools/gmm_evaluate.py \
$config \
$weight \
--eval bbox

# MultiCorrupt Evaluation
corruptions=("snow" "fog" "motionblur" "brightness" "missingcamera")
levels=("3" "2" "1")

for corruption in "${corruptions[@]}"; do
    for level in "${levels[@]}"; do
        python tools/gmm_evaluate.py \
        $config \
        $weight \
        --eval bbox \
        --overwrite_nuscenes_root=/workspace/multicorrupt/$corruption/$level
    done
done
```