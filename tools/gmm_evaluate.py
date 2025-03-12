# SPDX-License-Identifier: Apache-2.0
# 
# Copyright (c) 2025 Institute for Automotive Engineering of RWTH Aachen University
# Copyright (c) 2025 Computer Vision Group of RWTH Aachen University
# by Severin Heidrich, Till Beemelmanns, Alexey Nekrasov

import argparse
import os
import numpy as np
import torch
import warnings
import matplotlib.pyplot as plt
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import (init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor

from tools.gmm_utils import gmm_evaluate, entropy_prob
from netcal.metrics import ECE
from prettytable import PrettyTable


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
    
    feature_scale_lvl = args.feature_scale_lvl
    
    gmm = torch.load(
        os.path.join(os.path.dirname(args.checkpoint), f'train_gmm_scale_{feature_scale_lvl}.pt')
    )
    prior_log_prob = torch.load(
        os.path.join(os.path.dirname(args.checkpoint), f'train_prior_log_prob_scale_{feature_scale_lvl}.pt')
    )

    gmm_uncertainty_per_sample, softmax_entropy_per_sample, max_softmax_per_sample = gmm_evaluate(
        model=model,
        gmm=gmm,
        prior_log_prob=prior_log_prob,
        data_loader=data_loader,
        feature_scale_lvl=feature_scale_lvl,
    )

    output_dir = "clean" if args.overwrite_nuscenes_root is None else "_".join(args.overwrite_nuscenes_root.split("/")[-2:])
    output_dir += f"_scale{feature_scale_lvl}"
    save_dir = os.path.join(os.path.dirname(args.checkpoint), output_dir)
    os.makedirs(save_dir, exist_ok=True)

    np.savetxt(os.path.join(save_dir, 'gmm_uncertainty_per_sample.csv'), gmm_uncertainty_per_sample, delimiter=",")
    np.savetxt(os.path.join(save_dir, 'softmax_entropy_per_sample.csv'), softmax_entropy_per_sample, delimiter=",")
    np.savetxt(os.path.join(save_dir, 'max_softmax_per_sample.csv'), max_softmax_per_sample, delimiter=",")

    # Histogram of gmm_uncertainty_per_sample
    plt.figure()
    plt.hist(gmm_uncertainty_per_sample, bins=50)
    plt.xlabel("GMM Log Density")
    plt.ylabel("Frequency")
    plt.title("GMM Uncertainty per Sample")
    plt.savefig(os.path.join(save_dir, 'gmm_uncertainty_per_sample.png'))
    plt.close()

    # Histogram of softmax_entropy_per_sample
    plt.figure()
    plt.hist(softmax_entropy_per_sample, bins=50)
    plt.xlim(0, 0.2)
    plt.xlabel("Softmax Entropy")
    plt.ylabel("Frequency")
    plt.title("Softmax Entropy per Sample")
    plt.savefig(os.path.join(save_dir, 'softmax_entropy_per_sample.png'))
    plt.close()

    # Histogram of max_softmax_per_sample
    plt.figure()
    plt.hist(max_softmax_per_sample, bins=50)
    plt.xlim(0.8, 1.0)
    plt.xlabel("Max Softmax Probability")
    plt.ylabel("Frequency")
    plt.title("Max Softmax Probability per Sample")
    plt.savefig(os.path.join(save_dir, 'max_softmax_per_sample.png'))
    plt.close()

if __name__ == '__main__':
    main()
