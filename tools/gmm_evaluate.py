# SPDX-License-Identifier: Apache-2.0
# 
# Copyright (c) 2025 Institute for Automotive Engineering of RWTH Aachen University
# Copyright (c) 2025 Computer Vision Group of RWTH Aachen University
# by Severin Heidrich, Till Beemelmanns, Alexey Nekrasov
import argparse
import mmcv
import os
from tqdm import tqdm
import numpy as np
import torch
import warnings
import matplotlib.pyplot as plt
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.surroundocc.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor

from tools.gmm_utils import gmm_evaluate
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
from prettytable import PrettyTable

CLASS_NAMES = ['IoU','barrier','bicycle', 'bus', 'car', 'construction_vehicle',
               'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
               'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
               'manmade', 'vegetation']

INTERESTING_SAMPLES = ["n015-2018-07-11-11-54-16+0800__LIDAR_TOP__1531281624399157.pcd.bin"]

def count_parameters(model, print_table=True):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if print_table:
        print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate GMM Model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--is_vis',
        action='store_true',
        help='whether to generate output without evaluation.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--overwrite_nuscenes_root',
        type=str,
        default=None,
        help='overwrite the nuscenes root path in the config file')
    # paser for feature_scale_lvl
    parser.add_argument(
        '--feature_scale_lvl',
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help='feature scale level for GMM evaluation')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)


    # build the dataloader
    cfg.data.test['overwrite_nuscenes_root'] = args.overwrite_nuscenes_root
    # cfg.data.test["overwrite_nuscenes_root_for_cameras"] = ["CAM_FRONT"]

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    count_parameters(model, print_table=False)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if distributed:
        raise NotImplementedError
    
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    num_samples = len(data_loader)
    num_classes = 1 + len(dataset.class_names)
    device = torch.device('cpu')
    feature_scale_lvl = args.feature_scale_lvl
    
    wl = {
        3: 200,
        2: 100,
        1: 50,
        0: 25
    }[feature_scale_lvl]
    h = {
        3: 16,
        2: 8,
        1: 4,
        0: 2
    }[feature_scale_lvl]
    
    multivariate = torch.load(
        os.path.join(os.path.dirname(args.checkpoint), f'train_gmm_scale_{feature_scale_lvl}.pt'),
        map_location=device
    )
    
    gmm = torch.distributions.MultivariateNormal(
        loc=multivariate.loc,
        scale_tril=torch.linalg.cholesky(multivariate.covariance_matrix)
    )
    
    prior_log_prob = torch.load(
        os.path.join(os.path.dirname(args.checkpoint), f'train_prior_log_prob_scale_{feature_scale_lvl}.pt'),
        map_location=device
    )

    log_prob, logits, label_cls_ids, sample_names = gmm_evaluate(
        model=model,
        gmm=gmm,
        data_loader=data_loader,
        num_classes=num_classes,
        device=device,
        feature_scale_lvl=feature_scale_lvl,
    )

    # some values are -inf, set all -inf values to minimal finite number
    log_prob = torch.clamp(log_prob, min=-100000, max=100000)
    
    # log(z,y) + log(y)
    log_prob = log_prob + prior_log_prob
    
    # log q(z)
    gmm_uncertainty = torch.logsumexp(log_prob, dim=-1)
    gmm_uncertainty_per_sample = torch.mean(gmm_uncertainty, axis=-1)
    
    # mean log density per class
    gmm_log_density_mean_per_class = log_prob.mean(axis=(0, 1))
    
    softmax = torch.softmax(logits, dim=-1) # normal
    del logits # free memory
    
    softmax_entropy = entropy_prob(softmax) # get_confidence_from_entropy(entropy_prob(preds), num_classes)
    softmax_entropy_per_sample = torch.mean(softmax_entropy, axis=-1)

    # flatten all the tensors
    gmm_uncertainty = torch.flatten(gmm_uncertainty, 0, 1)
    softmax_entropy = torch.flatten(softmax_entropy, 0, 1)
    softmax = torch.flatten(softmax, 0, 1)
    log_prob = torch.flatten(log_prob, 0, 1)
    label_cls_ids = torch.flatten(label_cls_ids)
    
    pred_cls_ids = torch.max(softmax, dim=-1)[1]

    if not any(sample in sample_names for sample in INTERESTING_SAMPLES):
        save_indices = [0]
    else:
        save_indices = [sample_names.index(sample) for sample in INTERESTING_SAMPLES]

    pred_cls_ids_save = pred_cls_ids.view((num_samples, wl, wl, h))[save_indices].cpu().numpy()
    label_cls_ids_save = label_cls_ids.view((num_samples, wl, wl, h))[save_indices].cpu().numpy()
    softmax_entropy_save = softmax_entropy.view((num_samples, wl, wl, h))[save_indices].cpu().numpy()
    gmm_uncertainty_save = gmm_uncertainty.view((num_samples, wl, wl, h))[save_indices].cpu().numpy()
    sample_names_save = [sample_names[i] for i in save_indices]
    
    ##Save predictions and uncertainties for visualization
    output_dir = "clean" if args.overwrite_nuscenes_root is None else "_".join(args.overwrite_nuscenes_root.split("/")[-2:])
    output_dir += f"_scale{feature_scale_lvl}"
    save_dir = os.path.join(os.path.dirname(args.checkpoint), output_dir)
    
    for i in range(len(save_indices)):
        sample_save_dir = os.path.join(save_dir, sample_names_save[i])
        os.makedirs(sample_save_dir, exist_ok=True)
        np.save(os.path.join(sample_save_dir, 'predictions.npy'), pred_cls_ids_save[i])
        np.save(os.path.join(sample_save_dir, 'labels.npy'), label_cls_ids_save[i])
        np.save(os.path.join(sample_save_dir, 'entropy.npy'), softmax_entropy_save[i])
        np.save(os.path.join(sample_save_dir, 'gmm_log_density.npy'), gmm_uncertainty_save[i])
    del pred_cls_ids_save, label_cls_ids_save, softmax_entropy_save, gmm_uncertainty_save

    # Averages per Sample
    np.savetxt(os.path.join(save_dir, 'gmm_uncertainty_per_sample.csv'), gmm_uncertainty_per_sample.cpu().numpy(), delimiter=",")
    np.savetxt(os.path.join(save_dir, 'softmax_entropy_per_sample.csv'), softmax_entropy_per_sample.cpu().numpy(), delimiter=",")
    
    # gmm_uncertainty per voxel
    np.save(os.path.join(save_dir, 'gmm_uncertainty_per_voxel.npy'), gmm_uncertainty.cpu().numpy().astype(np.float16))
    
    # gmm_uncertainy per voxel only for occupied voxels
    mask = (pred_cls_ids != 0)
    np.save(os.path.join(save_dir, 'gmm_uncertainty_per_occupied_voxel.npy'), gmm_uncertainty[mask].cpu().numpy().astype(np.float16))

    # Histogram of gmm_uncertainty_per_sample
    plt.figure()
    plt.hist(gmm_uncertainty_per_sample.cpu().numpy(), bins=50)
    plt.xlabel("GMM Log Density")
    plt.ylabel("Frequency (normalized)")
    plt.title("GMM Uncertainty per Sample")
    plt.savefig(os.path.join(save_dir, 'gmm_uncertainty_per_sample.png'))
    plt.close()
    del gmm_uncertainty_per_sample, softmax_entropy_per_sample

    # Bar Plot of gmm_log_density_mean_per_class with class names
    class_names = np.array(['Unoccupied'] + dataset.class_names)
    plt.figure()
    plt.bar(class_names, gmm_log_density_mean_per_class.numpy())
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gmm_log_density_mean_per_class.png'))
    plt.close()

    # Print class_names, gmm_log_density_mean_per_class pretty format
    result = ""
    result += "Class Name: GMM Log Density Mean\n"
    for class_name, density_mean in zip(class_names, gmm_log_density_mean_per_class):
        result += f"{class_name}: {density_mean}\n"

    # Plot a normalized histogram of gmm_uncertainty for occupied voxels
    plt.figure()
    mask = (pred_cls_ids != 0)
    plt.hist(gmm_uncertainty[mask].numpy(), bins=50, density=True, range=(-50, 200))
    plt.xlabel("GMM Log Density")
    plt.ylabel("Frequency (normalized)")
    plt.title("GMM Uncertainty (pred_cls_ids != 0)")
    plt.savefig(os.path.join(save_dir, 'gmm_uncertainty_nonempty.png'))
    plt.close()


    # Plot a normalized histogram of gmm_uncertainty for empty
    plt.figure()
    mask = (pred_cls_ids == 0)
    plt.hist(gmm_uncertainty[mask].numpy(), bins=50, density=True, range=(-50, 200))
    plt.xlabel("GMM Log Density")
    plt.ylabel("Frequency (normalized)")
    plt.title("GMM Uncertainty (pred_cls_ids == 0)")
    plt.savefig(os.path.join(save_dir, 'gmm_uncertainty_empty.png'))
    plt.close()

    # Plot a normalized histogram of non-empty softmax_entropy
    plt.figure()
    mask = (pred_cls_ids != 0)
    plt.hist(softmax_entropy[mask].numpy(), bins=50, density=True, range=(0, 2))
    plt.xlabel("Softmax Entropy")
    plt.ylabel("Frequency (normalized)")
    plt.title("Softmax Entropy (pred_cls_ids != 0)")
    plt.savefig(os.path.join(save_dir, 'softmax_entropy_nonempty.png'))
    plt.close()
    
    # Plot a normalized histogram of empty softmax_entropy
    plt.figure()
    mask = (pred_cls_ids == 0)
    plt.hist(softmax_entropy[mask].numpy(), bins=50, density=True, range=(0, 2))
    plt.xlabel("Softmax Entropy")
    plt.ylabel("Frequency (normalized)")
    plt.title("Softmax Entropy (pred_cls_ids == 0)")
    plt.savefig(os.path.join(save_dir, 'softmax_entropy_empty.png'))
    plt.close()

    # Handle ignore class and cast to numpy
    mask = (label_cls_ids != 255)
    label_cls_ids = label_cls_ids[mask]
    pred_cls_ids = pred_cls_ids[mask]
    softmax = softmax[mask]
    softmax_entropy = softmax_entropy[mask]
    gmm_uncertainty = gmm_uncertainty[mask]

    # Calculate IoU, MIoU
    result += "\n\n"
    class_ious = []
    for j in range(len(CLASS_NAMES)):
        if j == 0: # class 0 for geometry IoU
            tp = ((label_cls_ids != 0) * (pred_cls_ids != 0)).sum() #TP
            tp_fn = (label_cls_ids != 0).sum() #TP+FN
            tp_fp = (pred_cls_ids != 0).sum() #TP+FP
        else:
            tp = ((label_cls_ids == j) * (pred_cls_ids == j)).sum() #TP
            tp_fn = (label_cls_ids == j).sum() #TP+FN
            tp_fp = (pred_cls_ids == j).sum() #TP+FP

        #Calculate the IoU for class j
        iou = tp/(tp_fn+tp_fp-tp) #TP/(TP+FN+FP)
        iou = iou.item()
        
        class_ious.append(iou) 
        result += CLASS_NAMES[j] + ":" + f"{iou:.4f}" + "\n"
    result += "mIoU:" + str(np.mean(np.array(class_ious)[1:])) + "\n\n"


    # Calculate NLL and ECE
    # NLL for Global Predictions
    nll = torch.nn.functional.nll_loss(
        input=torch.log(softmax + 1e-8),
        target=label_cls_ids.long(),
        reduction='mean'
    )
    result += f"Global NLL: {nll.item()}\n"

    # Calculate ECE
    n_bins = 10
    ece = ECE(n_bins)
    ece_score = ece.measure(
        X=softmax.max(axis=-1)[0].numpy(),
        y=(label_cls_ids==pred_cls_ids).numpy()
    )
    result += f"Global ECE: {ece_score}\n"

    # NLL Non-Empty Predictions
    mask = (pred_cls_ids != 0)
    non_empty_label_cls_ids = label_cls_ids[mask]
    non_empty_pred_cls_ids = pred_cls_ids[mask]
    non_empty_preds = softmax[mask]
    non_empty_softmax_entropy = softmax_entropy[mask]
    
    nll = torch.nn.functional.nll_loss(
        input=torch.log(non_empty_preds + 1e-8),
        target=non_empty_label_cls_ids.long(),
        reduction='mean'
    )
    result += f"Non-Empty NLL: {nll.item()}\n"
    
    ece_score = ece.measure(
        X=non_empty_preds.max(axis=-1)[0].numpy(),
        y=(non_empty_label_cls_ids==non_empty_pred_cls_ids).numpy()
    )
    result += f"Non-Empty ECE: {ece_score}\n"
    del non_empty_preds, non_empty_label_cls_ids, non_empty_pred_cls_ids, non_empty_softmax_entropy


    # NLL for Empty Predictions
    mask = (pred_cls_ids == 0)
    empty_label_cls_ids = label_cls_ids[mask]
    empty_pred_cls_ids = pred_cls_ids[mask]
    empty_preds = softmax[mask]
    empty_softmax_entropy = softmax_entropy[mask]
    
    nll = torch.nn.functional.nll_loss(
        input=torch.log(empty_preds + 1e-8),
        target=empty_label_cls_ids.long(),
        reduction='mean'
    )
    result += f"Empty NLL: {nll.item()}\n"
    
    ece_score = ece.measure(
        X=empty_preds.max(axis=-1)[0].numpy(),
        y=(empty_label_cls_ids==empty_pred_cls_ids).numpy()
    )
    result += f"Empty ECE: {ece_score}\n"
    del empty_preds, empty_label_cls_ids, empty_pred_cls_ids, empty_softmax_entropy
    
    
    print(result)
    with open(os.path.join(save_dir, 'results.txt'), 'w') as f:
        f.write(result)


def entropy_prob(probs):
    logp = torch.log(probs + 1e-12)
    plogp = probs * logp
    entropy = -torch.sum(plogp, axis=-1)
    return entropy


def get_confidence_from_entropy(entropy, class_num):
    #normalize entropy to have range [0,1] instead of range [0, log(class_num)]
    normalized_entropy = entropy / torch.log(torch.tensor(class_num))
    #invert normalized_entropy because low entropy means high confidence and vice versa
    inverted_normalized_entropy = 1 - normalized_entropy
    return inverted_normalized_entropy


if __name__ == '__main__':
    main()
