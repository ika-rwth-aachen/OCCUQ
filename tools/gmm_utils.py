# SPDX-License-Identifier: Apache-2.0
# 
# Copyright (c) 2025 Institute for Automotive Engineering of RWTH Aachen University
# Copyright (c) 2025 Computer Vision Group of RWTH Aachen University
# by Severin Heidrich, Till Beemelmanns, Alexey Nekrasov

import numpy as np
import torch
from tqdm import tqdm

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-308, 0, 1)]


def entropy_prob(probs):
    logp = torch.log(probs + 1e-12)
    plogp = probs * logp
    entropy = -torch.sum(plogp, axis=-1)
    return entropy


def gt_to_voxel(gt):
    voxel = torch.zeros((200, 200, 16), dtype=gt.dtype)
    voxel[gt[:, 0].long(), gt[:, 1].long(), gt[:, 2].long()] = gt[:, 3]
    return voxel


def multiscale_supervision(gt_occ, ratio, gt_shape):
    '''
    change ground truth shape as (B, W, H, Z) for each level supervision
    '''
    gt = torch.zeros([gt_shape[0], gt_shape[1], gt_shape[2], gt_shape[3]]).to(gt_occ.device).type(torch.float) 
    for i in range(gt.shape[0]):
        coords = gt_occ[i][:, :3].type(torch.long) // ratio
        gt[i, coords[:, 0], coords[:, 1], coords[:, 2]] =  gt_occ[i][:, 3]
    return gt


def get_features_balanced(
        model,
        data_loader,
        dtype,
        storage_device,
        max_features_per_class,
        max_collect_features_per_class_per_frame,
        feature_scale_lvl=3
    ):
    cls_names = ['unoccupied'] + data_loader.dataset.class_names
    cls_count = {} # Dictionary to keep count of features per class

    print("Max Features per Class: " + str(max_features_per_class))

    # Temporary lists to store features and labels
    features_list = []
    labels_list = []
    
    # Using tqdm to wrap the data_loader for progress bar
    progress_bar = tqdm(data_loader, desc="Processing", leave=True, dynamic_ncols=True)

    for _, data in enumerate(progress_bar):
        with torch.no_grad():

            _ = model(return_loss=False, rescale=True, **data)
            output = model.module.pts_bbox_head.feature
            
            pred = model.module.pred.permute(0, 2, 3, 4, 1)
            pred_ids = pred.max(-1)[1].to(device=storage_device)
            gt_occ = data['gt_occ'][0]
            
            scale_ratio = 2**(len(output) - 1 - feature_scale_lvl)
            label = multiscale_supervision(gt_occ.clone().unsqueeze(0), scale_ratio, output[feature_scale_lvl].shape).squeeze(0)
            feature = output[feature_scale_lvl].squeeze(0)

            # Iterate over each class in the label
            for idx, cls_name in enumerate(cls_names):
                if cls_name not in cls_count:
                    cls_count[cls_name] = 0

                # Check if the limit for this class is reached
                if cls_count[cls_name] < max_features_per_class:
                    #class_filter = ((pred_ids == label) & (label == idx)).squeeze() # TP Filter
                    class_filter = (label == idx).squeeze()
                    
                    #print("Total", (label == idx).sum())
                    #print("TP", ((label == pred_ids) & (label == idx)).sum())
                    #print("FP", ((label != pred_ids) & (label == idx)).sum())
                    
                    new_count_candidate = cls_count[cls_name] + torch.sum(class_filter)
                    
                    if new_count_candidate > max_features_per_class:
                        num_features_to_add = max_features_per_class - cls_count[cls_name]
                    else:
                        num_features_to_add = torch.sum(class_filter)
                        num_features_to_add = torch.clamp(num_features_to_add, max=max_collect_features_per_class_per_frame)
                    
                    feature_per_label = feature[class_filter].to(dtype=dtype, device=storage_device)
                    label_per_feature = label[class_filter].to(dtype=torch.uint8, device=storage_device)
                    
                    # Shuffle the features to add
                    feature_idx = np.arange(feature_per_label.shape[0])
                    np.random.shuffle(feature_idx)
                    feature_idx = feature_idx[:num_features_to_add]
                    
                    feature_per_label = feature_per_label[feature_idx, ...]
                    label_per_feature = label_per_feature[feature_idx, ...]
                    
                    features_list.append(feature_per_label)
                    labels_list.append(label_per_feature)
                    
                    cls_count[cls_name] += feature_per_label.size(0)

            # Stop processing if we have enough features for each class
            if all(count >= max_features_per_class for count in cls_count.values()):
                break
        
        # Update the progress bar with class counts
        class_counts_str = ', '.join([f'{class_name}: {count}' for class_name, count in cls_count.items()])
        progress_bar.set_postfix_str(f'Class counts: {class_counts_str}')

    # Ensure the progress bar is closed properly
    progress_bar.close()
    
    class_counts_str = '\n'.join([f'Class {class_name}: {count}' for class_name, count in cls_count.items()])
    print(class_counts_str)

    features_list = torch.cat(features_list)
    labels_list = torch.cat(labels_list)

    return features_list, labels_list


def gmm_fit(features, labels, num_classes):
    #ignore the ignore class
    mask = (labels != 255)
    labels = labels[mask]
    features = features[mask]

    # mean computation
    classwise_mean_features = []
    for c in tqdm(range(num_classes)):
        cls_features = features[labels == c]
        if len(cls_features) > 0:
            mean_feature = torch.mean(cls_features, dim=0)
        else:
            mean_feature = torch.zeros(features.shape[1], device=features.device)
        classwise_mean_features.append(mean_feature)
    classwise_mean_features = torch.stack(classwise_mean_features)

    # covariance computation
    classwise_cov_features = []
    for c in tqdm(range(num_classes)):
        cls_features = features[labels == c].t()
        if cls_features.shape[1] > 0:
            cov_feature = torch.cov(cls_features)
        else:
            cov_feature = torch.eye(features.shape[1], device=features.device)
        classwise_cov_features.append(cov_feature)
    classwise_cov_features = torch.stack(classwise_cov_features)

    gmm = None
    with torch.no_grad():
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(classwise_cov_features.shape[1], device=classwise_cov_features.device).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=classwise_mean_features, covariance_matrix=(classwise_cov_features + jitter),
                )
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue
            except ValueError as e:
                if "The parameter covariance_matrix has invalid values" in str(e):
                    continue
            if gmm is not None:
                break
    
    # Check if gmm is still None after the loop
    if gmm is None:
        raise RuntimeError("Failed to create a valid MultivariateNormal distribution with the provided jitter values.")

    print("Fitted GMM with JITTER EPS= ", jitter_eps)
    
    return gmm


def get_prior_log_prob(labels, num_classes):
    class_counts = torch.bincount(labels, minlength=num_classes)
    total_samples = labels.size(0)
    return torch.log(class_counts.float() / total_samples)


def gmm_evaluate(model, gmm, data_loader, num_classes, device, feature_scale_lvl=3):
    num_voxels_per_scale = {
        3: 200*200*16,
        2: 100*100*8,
        1: 50*50*4,
        0: 25*25*2,
    }[feature_scale_lvl]
    
    num_samples = len(data_loader.dataset)

    log_probs = torch.empty((num_samples, num_voxels_per_scale, num_classes), dtype=torch.float32, device=device)
    logits = torch.empty((num_samples, num_voxels_per_scale, num_classes), dtype=torch.float32, device=device)
    labels = torch.empty((num_samples, num_voxels_per_scale), dtype=torch.int, device=device)
    sample_name = []

    for i, data in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **data)

            feature = model.module.pts_bbox_head.feature[feature_scale_lvl]
            logit = model.module.logits[feature_scale_lvl].permute(0, 2, 3, 4, 1)

            scale_ratio = 2**(len(model.module.logits) - 1 - feature_scale_lvl)
            label = multiscale_supervision(data['gt_occ'].clone(), scale_ratio, feature.shape)

            feature = torch.flatten(feature, 0, 3).to(device)
            label = torch.flatten(label).to(device)
            logit = torch.flatten(logit, 0, 3).to(device)
            
            log_prob = gmm.log_prob(feature[:, None, :])

            log_probs[i, :, :] = log_prob
            logits[i, :, :] = logit
            labels[i, :] = label
            
            sample_name.append(data['img_metas'].data[0][0]['occ_path'].replace('.npy', '').split('/')[-1])

    return log_probs, logits, labels, sample_name

