<!-- omit in toc -->
# RUN OCCUQ

<!-- omit in toc -->
# Outline
- [Download trained models and GMM](#download-trained-models-and-gmm)
- [Train \& Test OCCUQ](#train--test-occuq)
  - [Train OCCUQ Model with 4 A100 GPUs](#train-occuq-model-with-4-a100-gpus)
  - [Train GMM with 1 A100 GPU](#train-gmm-with-1-a100-gpu)
  - [Evaluate OCCUQ with 1 A100 GPU](#evaluate-occuq-with-1-a100-gpu)
  - [Video Generation](#video-generation)
  - [MultiCorrupt Evaluation](#multicorrupt-evaluation)


## Download trained models and GMM
Download and unzip weights and GMM to `work_dirs` to obtain the following structure:

[Download Link](https://rwth-aachen.sciebo.de/s/2o1LOb4PwFbPzSb)


```
work_dirs
├── occuq_mlpv5_sn
│   ├── 20240821_225901.log
│   ├── 20240821_225901.log.json
│   ├── epoch_6.pth
│   ├── occuq_mlpv5_sn.py.py
│   ├── train_gmm_scale_0.pt
│   ├── train_gmm_scale_1.pt
│   ├── train_gmm_scale_2.pt
│   ├── train_gmm_scale_3.pt
│   ├── train_prior_log_prob_scale_0.pt
│   ├── train_prior_log_prob_scale_1.pt
│   ├── train_prior_log_prob_scale_2.pt
│   └── train_prior_log_prob_scale_3.pt
```


## Train & Test OCCUQ
Check folder `/workspace/scripts` for some scripts to train and test the models.


### Train OCCUQ Model with 4 A100 GPUs
```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,4
export PYTHONPATH=$PYTHONPATH:/workspace

./tools/dist_train.sh \
/workspace/projects/configs/occuq/occuq_mlpv5_sn.py 4 \
/workspace/work_dirs/occuq_mlpv5_sn
```


### Train GMM with 1 A100 GPU
```bash
python tools/gmm_fit.py \
$config \
$weight \
--eval bbox
```


### Evaluate OCCUQ with 1 A100 GPU
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

### Video Generation
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

### MultiCorrupt Evaluation
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