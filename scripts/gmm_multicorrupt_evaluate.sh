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
levels=("1" "2" "3")

for corruption in "${corruptions[@]}"; do
    for level in "${levels[@]}"; do
        python tools/gmm_evaluate.py \
        $config \
        $weight \
        --eval bbox \
        --overwrite_nuscenes_root=/workspace/multicorrupt/$corruption/$level
    done
done

python scripts/ood_detection_evaluation.py \
--work_dirs /workspace/work_dirs/occuq_mlpv5_sn


# Mean across all corruptions and levels
#         Measure  mAUROC  mAUPR  mFPR95
#             GMM   80.15  79.43   56.18
# Softmax Entropy   54.63  56.21   94.47
#     Max Softmax   56.16  57.52   93.17

#         Measure  snow_1_AUROC  snow_1_AUPR  snow_1_FPR95
#             GMM         62.84        62.28         85.58
# Softmax Entropy         50.87        52.00         95.40
#     Max Softmax         51.20        52.89         95.68

#         Measure  snow_2_AUROC  snow_2_AUPR  snow_2_FPR95
#             GMM         91.30        91.53         39.41
# Softmax Entropy         51.44        56.50         99.12
#     Max Softmax         53.70        58.56         98.60

#         Measure  snow_3_AUROC  snow_3_AUPR  snow_3_FPR95
#             GMM         99.85        99.86          0.30
# Softmax Entropy         82.81        85.88         79.10
#     Max Softmax         85.98        88.17         68.72

#         Measure  fog_1_AUROC  fog_1_AUPR  fog_1_FPR95
#             GMM        61.81       60.44        86.01
# Softmax Entropy        52.61       54.28        93.97
#     Max Softmax        53.03       54.77        93.89

#         Measure  fog_2_AUROC  fog_2_AUPR  fog_2_FPR95
#             GMM        72.39       71.91        77.62
# Softmax Entropy        56.62       59.39        94.22
#     Max Softmax        57.61       60.13        93.79

#         Measure  fog_3_AUROC  fog_3_AUPR  fog_3_FPR95
#             GMM        91.47       91.92        41.09
# Softmax Entropy        69.27       73.36        90.78
#     Max Softmax        71.15       74.82        88.97

#         Measure  motionblur_1_AUROC  motionblur_1_AUPR  motionblur_1_FPR95
#             GMM               69.25              68.81               79.61
# Softmax Entropy               46.75              47.61               96.25
#     Max Softmax               47.70              48.45               96.06

#         Measure  motionblur_2_AUROC  motionblur_2_AUPR  motionblur_2_FPR95
#             GMM               82.76              82.26               59.35
# Softmax Entropy               45.16              46.46               98.50
#     Max Softmax               46.94              47.77               97.99

#         Measure  motionblur_3_AUROC  motionblur_3_AUPR  motionblur_3_FPR95
#             GMM               90.92              90.88               40.21
# Softmax Entropy               49.38              50.71               96.84
#     Max Softmax               51.74              52.58               95.50

#         Measure  brightness_1_AUROC  brightness_1_AUPR  brightness_1_FPR95
#             GMM               60.96              60.10               87.49
# Softmax Entropy               49.80              49.23               95.30
#     Max Softmax               50.06              49.35               95.40

#         Measure  brightness_2_AUROC  brightness_2_AUPR  brightness_2_FPR95
#             GMM               72.35              72.57               78.73
# Softmax Entropy               50.10              49.99               97.47
#     Max Softmax               50.78              50.41               97.39

#         Measure  brightness_3_AUROC  brightness_3_AUPR  brightness_3_FPR95
#             GMM               91.79              92.29               37.98
# Softmax Entropy               54.95              55.75               99.67
#     Max Softmax               56.67              56.79               99.37

#         Measure  missingcamera_1_AUROC  missingcamera_1_AUPR  missingcamera_1_FPR95
#             GMM                  71.80                 67.67                  70.51
# Softmax Entropy                  53.94                 53.75                  93.11
#     Max Softmax                  54.85                 54.40                  92.39

#         Measure  missingcamera_2_AUROC  missingcamera_2_AUPR  missingcamera_2_FPR95
#             GMM                  87.32                 84.34                  40.77
# Softmax Entropy                  54.98                 55.71                  92.57
#     Max Softmax                  56.81                 57.33                  90.93

#         Measure  missingcamera_3_AUROC  missingcamera_3_AUPR  missingcamera_3_FPR95
#             GMM                  95.39                 94.54                  18.03
# Softmax Entropy                  50.82                 52.49                  94.78
#     Max Softmax                  54.22                 56.36                  92.91

