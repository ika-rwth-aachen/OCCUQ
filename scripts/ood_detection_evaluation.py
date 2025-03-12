import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, auc
import os
import pandas as pd
import argparse


def calculate_auroc(predictions, targets):
    fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=predictions)
    roc_auc = auc(fpr, tpr)
    fpr_best = 0
    threshold = 0
    for i, j, threshold in zip(tpr, fpr, thresholds):
        if i > 0.95:
            fpr_best = j
            break
    return roc_auc, fpr_best, threshold


def main(work_dirs, feature_scale_lvl):
    corruption_types = ["snow", "fog", "motionblur", "brightness", "missingcamera"]
    corruptions = [f"{corruption}_{i}" for corruption in corruption_types for i in range(1, 4)]
    measures = ["gmm_uncertainty_per_sample", "softmax_entropy_per_sample", "max_softmax_per_sample"]
    nice_measure_names = ["GMM", "Softmax Entropy", "Max Softmax"]
    scale = f"_scale{feature_scale_lvl}"
    all_corruptions_available = True

    results = []
    for measure in measures:
        measure_results = {"Measure": nice_measure_names[measures.index(measure)]}
        
        # id data
        id_file = f'{work_dirs}/clean{scale}/{measure}.csv'
        id_density = np.loadtxt(id_file)
        id_labels = np.ones(id_density.shape[0])
        
        for corruption in corruptions:
            ood_file = f'{work_dirs}/{corruption}{scale}/{measure}.csv'
            
            if not os.path.isfile(ood_file):
                all_corruptions_available = False
                continue
            
            ood_density = np.loadtxt(ood_file)
            ood_labels = np.zeros(ood_density.shape[0])
            
            labels = np.concatenate((id_labels, ood_labels))
            scores = np.concatenate((id_density, ood_density))
            
            if measure == "max_softmax_per_sample":
                scores = 1 - scores
            
            auroc, fpr_best, _ = calculate_auroc(scores, labels)
            aupr = average_precision_score(labels, scores)
            
            measure_results[f"{corruption}_AUROC"] = auroc * 100
            measure_results[f"{corruption}_AUPR"] = aupr * 100
            measure_results[f"{corruption}_FPR95"] = fpr_best * 100
        
        results.append(measure_results)

    df = pd.DataFrame(results)
    for corruption in corruptions:
        print(df[["Measure", f"{corruption}_AUROC", f"{corruption}_AUPR", f"{corruption}_FPR95"]].round(2).to_string(index=False))
        print()

    if all_corruptions_available:
        df["mAUROC"] = df[[f"{corruption}_AUROC" for corruption in corruptions]].mean(axis=1)
        df["mAUPR"] = df[[f"{corruption}_AUPR" for corruption in corruptions]].mean(axis=1)
        df["mFPR95"] = df[[f"{corruption}_FPR95" for corruption in corruptions]].mean(axis=1)

        print(df[["Measure", "mAUROC", "mAUPR", "mFPR95"]].round(2).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OOD Detection Evaluation')
    parser.add_argument(
        '--work_dirs',
        type=str,
        default='/workspace/work_dirs/occuq_mlpv5_sn',
        help='Working directories'
    )
    parser.add_argument(
        '--feature_scale_lvl',
        type=int,
        default=3,
        help='Feature scale level'
    )
    args = parser.parse_args()
    main(args.work_dirs, args.feature_scale_lvl)
