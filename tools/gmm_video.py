#  Copyright (c) Institute for Automotive Engineering of RWTH Aachen University
#  Copyright (c) Visual Computing Institute of RWTH Aachen University
#  by Severin Heidrich, Till Beemelmanns, Alexey Nekrasov
# ---------------------------------------------

import argparse
import os
import torch
import warnings
import numpy as np
from tqdm import tqdm

from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import (load_checkpoint, wrap_fp16_model)

from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor

from prettytable import PrettyTable

import cv2
from PIL import Image
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from open3d import geometry
import matplotlib.cm as cm
from scipy.signal import savgol_filter
import glob
import os
import io
from PIL import Image

matplotlib.use('Agg')


def entropy_prob(probs):
    logp = torch.log(probs + 1e-12)
    plogp = probs * logp
    entropy = -torch.sum(plogp, axis=-1)
    return entropy


def multiscale_supervision(gt_occ, ratio, gt_shape):
    '''
    change ground truth shape as (B, W, H, Z) for each level supervision
    '''
    gt = torch.zeros([gt_shape[0], gt_shape[1], gt_shape[2], gt_shape[3]]).to(gt_occ.device).type(torch.float) 
    for i in range(gt.shape[0]):
        coords = gt_occ[i][:, :3].type(torch.long) // ratio
        gt[i, coords[:, 0], coords[:, 1], coords[:, 2]] =  gt_occ[i][:, 3]
    return gt


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
        description='MMDet test (and eval) a model')
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

COLOR_MAP = np.array(
    [
        [0, 0, 0, 255], # empty
        [255, 120, 50, 255],  # barrier              orangey
        [255, 192, 203, 255],  # bicycle              pink
        [255, 255, 0, 255],  # bus                  yellow
        [0, 150, 245, 255],  # car                  blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [200, 180, 0, 255],  # motorcycle           dark orange
        [255, 0, 0, 255],  # pedestrian           red
        [255, 240, 150, 255],  # traffic_cone         light yellow
        [135, 60, 0, 255],  # trailer              brown
        [160, 32, 240, 255],  # truck                purple
        [255, 0, 255, 255],  # driveable_surface    dark pink
        # [175,   0,  75, 255],       # other_flat           dark red
        [139, 137, 137, 255], # other flat
        [75, 0, 75, 255],  # sidewalk             dard purple
        [150, 240, 80, 255],  # terrain              light green
        [230, 230, 250, 255],  # manmade              white
        [0, 175, 0, 255],  # vegetation           green
    ]
) / 255.


def get_pointcloud(predictions, uncertainties, vmin, vmax, cmap="RdYlGn"):
    predictions = predictions.squeeze()
    uncertainties = uncertainties.squeeze()

    scalar_mappable = cm.ScalarMappable(cmap=cmap)
    scalar_mappable.set_clim(vmin=vmin, vmax=vmax)

    pcl_range = [-50, -50, -5.0, 50, 50, 3.0]
    occ_size = [200, 200, 16]

    x = np.linspace(0, predictions.shape[0] - 1, predictions.shape[0]) #x.shape = torch.Size([200])
    y = np.linspace(0, predictions.shape[1] - 1, predictions.shape[1]) #y.shape = torch.Size([200])
    z = np.linspace(0, predictions.shape[2] - 1, predictions.shape[2]) #z.shape = torch.Size([16])
    X, Y, Z = np.meshgrid(x, y, z) #X.shape = Y.shape = Z.shape = torch.Size([200, 200, 16])
    vv = np.stack([X, Y, Z], axis=-1) #vv.shape = torch.Size([200, 200, 16, 3])

    vertices = vv[predictions > 0.5] #vertices.shape = torch.Size([61753, 3])
    vertices[:, 0] = (vertices[:, 0] + 0.5) * (pcl_range[3] - pcl_range[0]) /  occ_size[0]  + pcl_range[0]
    vertices[:, 1] = (vertices[:, 1] + 0.5) * (pcl_range[4] - pcl_range[1]) /  occ_size[1]  + pcl_range[1]
    vertices[:, 2] = (vertices[:, 2] + 0.5) * (pcl_range[5] - pcl_range[2]) /  occ_size[2]  + pcl_range[2]
    #vertices contains the (x,y,z) coordinates of all voxels with class other than 0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    uncertainties = uncertainties[predictions > 0]

    # Get colors for the uncertainties
    uncertainty_colors = scalar_mappable.to_rgba(uncertainties)

    pcd.colors = o3d.utility.Vector3dVector(uncertainty_colors[..., :3])
    return pcd


def plot_pointcloud(point_cloud, elev=50, azim=30, cmap="RdYlGn", file_path=None):
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    # Create a matplotlib figure with higher DPI
    fig = plt.figure(figsize=(8, 12), dpi=300)

    # Add a 3D axis without ticks and labels
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')

    # Set view angle (more from the side)
    ax.view_init(elev=elev, azim=azim)

    # Scatter plot with colors and smaller point size
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        cmap=cmap,
        marker='.',
        s=5,
        alpha=1.0
    )
    
    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 35)
    ax.set_zlim(-10, 10)
    
    #ax.text2D(0.5, 0.95, title, ha='center', va='center', transform=ax.transAxes, fontsize=14)
    if file_path is not None:
        fig.savefig(file_path, bbox_inches='tight', transparent=False)
        return None
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=False)
        buf.seek(0)
        image = Image.open(buf)
        return image


def get_label_pointcloud(labels):
    labels = labels.squeeze()
    pcl_range = [-50, -50, -5.0, 50, 50, 3.0]
    occ_size = [200, 200, 16]

    x = np.linspace(0, labels.shape[0] - 1, labels.shape[0]) #x.shape = torch.Size([200])
    y = np.linspace(0, labels.shape[1] - 1, labels.shape[1]) #y.shape = torch.Size([200])
    z = np.linspace(0, labels.shape[2] - 1, labels.shape[2]) #z.shape = torch.Size([16])
    X, Y, Z = np.meshgrid(x, y, z) #X.shape = Y.shape = Z.shape = torch.Size([200, 200, 16])
    vv = np.stack([X, Y, Z], axis=-1) #vv.shape = torch.Size([200, 200, 16, 3])

    vertices = vv[(labels > 0.5) & (labels != 255)] #vertices.shape = torch.Size([61753, 3])
    vertices[:, 0] = (vertices[:, 0] + 0.5) * (pcl_range[3] - pcl_range[0]) /  occ_size[0]  + pcl_range[0]
    vertices[:, 1] = (vertices[:, 1] + 0.5) * (pcl_range[4] - pcl_range[1]) /  occ_size[1]  + pcl_range[1]
    vertices[:, 2] = (vertices[:, 2] + 0.5) * (pcl_range[5] - pcl_range[2]) /  occ_size[2]  + pcl_range[2]
    #vertices contains the (x,y,z) coordinates of all voxels with class other than 0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    # Filter labels and predictions
    filtered_labels = labels[(labels > 0) & (labels != 255)]
    
    # Initialize color array to black
    colors = np.zeros((len(vertices), 3))
    colors = [(COLOR_MAP[i, :3]/255.0) for i in filtered_labels]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
    
    
def plot_label_pointcloud(point_cloud, title, file_path=None, plot_legend=False):
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    # Create a matplotlib figure with higher DPI
    fig = plt.figure(figsize=(8, 12), dpi=300)

    # Add a 3D axis without ticks and labels
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')

    # Set view angle (more from the side)
    ax.view_init(elev=50, azim=30)

    # Scatter plot with colors and smaller point size
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, marker='.', s=5, alpha=1)

    # Adjust aspect ratio
    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 35)
    ax.set_zlim(-10, 10)
    
    # Annotate title directly onto the plot
    # ax.text2D(0.5, 0.85, title, ha='center', va='center', transform=ax.transAxes, fontsize=14)
    if plot_legend:
        class_names = [
            "empty", "Barrier", "Bicycle", "Bus", "Car", "Construction Vehicle",
            "Motorcycle", "Pedestrian", "Traffic Cone", "Trailer", "Truck",
            "Driveable Surface", "Other Flat", "Sidewalk", "Terrain", "Manmade",
            "Vegetation"
        ]

        legend_entries = []
        for class_name, color in zip(class_names[1:], COLOR_MAP[1:]):
            legend_entries.append(mpatches.Patch(color=color, label=class_name))
        plt.legend(handles=legend_entries, loc="best", bbox_to_anchor=(0.45, 0.6), ncol=2, fontsize="x-small", frameon=True)

    if file_path is not None:
        fig.savefig(file_path, bbox_inches='tight', transparent=False)
        return None
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=False)
        buf.seek(0)
        image = Image.open(buf)
        return image


def gmm_video(model, gmm, prior_log_prob, data_loader, feature_scale_lvl, num_samples, save_dir):
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
    
    gmm_mean_values = []
    entropy_mean_values = []
    for i, data in tqdm(enumerate(data_loader), total=num_samples):
        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **data)
            images = data['img'].data[0].squeeze().permute(0, 2, 3, 1)
            images = images[:, :900, :, :] # remove padding
            images = images + torch.tensor([103.530, 116.280, 123.675])
            images = images[..., [2, 1, 0]]
            images = torch.clamp(images, 0, 255)
            
            images = images.cpu().numpy().astype(np.uint8)
            
            total_width = images.shape[2] * 3
            total_height = images.shape[1] * 2
            new_image = Image.new('RGB', (total_width, total_height))
            x_offset = 0
            y_offset = 0
            for counter, camera_id in enumerate([2, 0, 1, 5, 3 ,4]):
                img = Image.fromarray(images[camera_id])
                new_image.paste(img, (x_offset, y_offset))
                x_offset += img.width
                
                if (counter + 1) % 3 == 0:
                    x_offset = 0
                    y_offset += img.height
            
            new_image = new_image.resize((total_width // 2, total_height // 2))
            new_image.save(f"{save_dir}/cameras_{str(i).zfill(4)}.jpg")
            
            feature = model.module.pts_bbox_head.feature[feature_scale_lvl]
            logit = model.module.logits[feature_scale_lvl].permute(0, 2, 3, 4, 1)

            scale_ratio = 2**(len(model.module.logits) - 1 - feature_scale_lvl)
            label = multiscale_supervision(data['gt_occ'].clone(), scale_ratio, feature.shape)

            feature = torch.flatten(feature, 0, 3).cpu()
            label = torch.flatten(label).cpu()
            logit = torch.flatten(logit, 0, 3).cpu()
            
            log_prob = gmm.log_prob(feature[:, None, :])
            log_prob = torch.clamp(log_prob, min=-100000, max=100000)
            log_prob = log_prob + prior_log_prob
            
            gmm_uncertainty = torch.logsumexp(log_prob, dim=-1)
            gmm_uncertainty = gmm_uncertainty.view(1, wl, wl, h).cpu().numpy()
            
            softmax = torch.softmax(logit, dim=-1)
            pred_cls_ids = torch.max(softmax, dim=-1)[1]
            pred_cls_ids = pred_cls_ids.view(1, wl, wl, h).cpu().numpy()

            gmm_densities_pc = get_pointcloud(
                pred_cls_ids,
                gmm_uncertainty,
                vim=-90,
                vmax=-20,
            )
            plot_pointcloud(
                gmm_densities_pc,
                file_path=f"{save_dir}/gmm_density_" + str(i).zfill(4) + ".png"
            )


            prediction_pc = get_label_pointcloud(pred_cls_ids)
            plot_label_pointcloud(
                prediction_pc,
                "Prediction",
                file_path=f"{save_dir}/prediction_" + str(i).zfill(4) + ".png"
            )

            # Smooth gmm_mean_values using a moving average
            gmm_mean_values.append(gmm_uncertainty.mean())
            smoothed_value = savgol_filter(gmm_mean_values, window_length=3, polyorder=2, mode='nearest')
            smoothed_value = -gmm_uncertainty.mean() # -smoothed_value
            vmin = 10
            vmax = 15
            smoothed_value = (smoothed_value - vmin) / (vmax - vmin)
            # Plot a single bar plot
            plt.figure(figsize=(3, 8))
            plt.bar(0, height=smoothed_value)
            plt.ylim(0, 1)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.savefig(f'{save_dir}/gmm_smooth_' + str(i).zfill(4) + '.png')
            plt.close()
            
            
            entropy = entropy_prob(softmax).mean().cpu().numpy()
            entropy_mean_values.append(entropy)
            smoothed_value = entropy # savgol_filter(entropy_mean_values, window_length=3, polyorder=2, mode='nearest')
            vmin = 0.0
            vmax = 0.25
            smoothed_value = (smoothed_value - vmin) / (vmax - vmin)
            plt.figure(figsize=(3, 8))
            plt.bar(0, height=smoothed_value)
            plt.ylim(0, 1)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.savefig(f'{save_dir}/entropy_smooth_' + str(i).zfill(4) + '.png')
            plt.close()

            if i == num_samples - 1:
                break

    # create video
    for pattern in [f'{save_dir}/cameras_*.jpg',
                    f'{save_dir}/gmm_density_*.png',
                    f'{save_dir}/prediction_*.png',
                    f'{save_dir}/gmm_smooth_*.png',
                    f'{save_dir}/entropy_smooth_*.png']:
        video_name = pattern.replace('*', 'video').replace('.jpg', '.mp4').replace('.png', '.mp4')

        # Get the list of image files in the folder
        image_files = sorted(glob.glob(pattern))

        # Read the first image to get the dimensions
        first_image = cv2.imread(image_files[0])
        height, width, _ = first_image.shape

        # Define the video codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, 6, (width, height))

        # Iterate over the image files and write each frame to the video
        for image_file in image_files:
            frame = cv2.imread(image_file)
            video.write(frame)
            os.remove(image_file)

        # Release the VideoWriter object and close the video file
        video.release()
        cv2.destroyAllWindows()


def gmm_video_v2(model, gmm, prior_log_prob, data_loader, feature_scale_lvl, num_samples, save_dir):
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
    
    gmm_mean_values = []
    entropy_mean_values = []
    for i, data in tqdm(enumerate(data_loader), total=num_samples):
        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **data)
            images = data['img'].data[0].squeeze().permute(0, 2, 3, 1)
            images = images[:, :900, :, :] # remove padding
            images = images + torch.tensor([103.530, 116.280, 123.675])
            images = images[..., [2, 1, 0]]
            images = torch.clamp(images, 0, 255)
            
            images = images.cpu().numpy().astype(np.uint8)
            
            total_width = images.shape[2] * 3
            total_height = images.shape[1] * 2
            new_image = Image.new('RGB', (total_width, total_height))
            x_offset = 0
            y_offset = 0
            for counter, camera_id in enumerate([2, 0, 1, 5, 3 ,4]):
                img = Image.fromarray(images[camera_id])
                new_image.paste(img, (x_offset, y_offset))
                x_offset += img.width
                
                if (counter + 1) % 3 == 0:
                    x_offset = 0
                    y_offset += img.height
            
            new_image = new_image.resize((total_width // 2, total_height // 2))
            
            feature = model.module.pts_bbox_head.feature[feature_scale_lvl]
            logit = model.module.logits[feature_scale_lvl].permute(0, 2, 3, 4, 1)

            scale_ratio = 2**(len(model.module.logits) - 1 - feature_scale_lvl)
            label = multiscale_supervision(data['gt_occ'].clone(), scale_ratio, feature.shape)

            feature = torch.flatten(feature, 0, 3).cpu()
            label = torch.flatten(label).cpu()
            logit = torch.flatten(logit, 0, 3).cpu()
            
            log_prob = gmm.log_prob(feature[:, None, :])
            log_prob = torch.clamp(log_prob, min=-100000, max=100000)
            log_prob = log_prob + prior_log_prob
            
            gmm_uncertainty = torch.logsumexp(log_prob, dim=-1)
            gmm_uncertainty = gmm_uncertainty.view(1, wl, wl, h).cpu().numpy()
            
            softmax = torch.softmax(logit, dim=-1)
            pred_cls_ids = torch.max(softmax, dim=-1)[1]
            pred_cls_ids = pred_cls_ids.view(1, wl, wl, h).cpu().numpy()

            gmm_uncertainty_pc = get_pointcloud(
                pred_cls_ids,
                gmm_uncertainty,
                vmin=-90,
                vmax=-20
            )
            uncertainty_img = plot_pointcloud(
                gmm_uncertainty_pc,
                elev=50,
                azim=30,
            )

            uncertainty_img2 = plot_pointcloud(
                gmm_uncertainty_pc,
                elev=5,
                azim=0,
            )

            prediction_pc = get_label_pointcloud(pred_cls_ids)
            prediction_img = plot_label_pointcloud(
                prediction_pc,
                "Prediction"
            )
            width, height = uncertainty_img.size
            uncertainty_img = uncertainty_img.crop((100, 200, width, height - 300))
            prediction_img = prediction_img.crop((100, 200, width, height - 300))
            uncertainty_img2 = uncertainty_img2.crop((100, 200, width, height - 300))
            
            uncertainty_img = uncertainty_img.resize((uncertainty_img.width // 2, uncertainty_img.height // 2))
            prediction_img = prediction_img.resize((prediction_img.width // 2, prediction_img.height // 2))
            uncertainty_img2 = uncertainty_img2.resize((uncertainty_img2.width // 2, uncertainty_img2.height // 2))
            
            # Combine all images into one
            composite_image = Image.new('RGB', (new_image.width, new_image.height + uncertainty_img.height))
            composite_image.paste(new_image, (0, 0))
            composite_image.paste(uncertainty_img, (0, new_image.height))
            composite_image.paste(uncertainty_img2, (uncertainty_img.width, new_image.height))
            composite_image.paste(prediction_img, (uncertainty_img.width * 2, new_image.height))

            composite_image.save(f"{save_dir}/composite_{str(i).zfill(4)}.jpg")

            if i == num_samples - 1:
                break

    # create video
    video_name = f'{save_dir}/composite_video.mp4'
    image_files = sorted(glob.glob(f'{save_dir}/composite_*.jpg'))

    # Read the first image to get the dimensions
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # Define the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 6, (width, height))

    # Iterate over the image files and write each frame to the video
    for image_file in image_files:
        frame = cv2.imread(image_file)
        video.write(frame)
        os.remove(image_file)

    # Release the VideoWriter object and close the video file
    video.release()
    cv2.destroyAllWindows()



def gmm_video_v3(model, gmm, prior_log_prob, data_loader, feature_scale_lvl, num_samples, save_dir):
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

    for i, data in tqdm(enumerate(data_loader), total=num_samples):
        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **data)
            images = data['img'].data[0].squeeze().permute(0, 2, 3, 1)
            images = images[:, :900, :, :] # remove padding
            images = images + torch.tensor([103.530, 116.280, 123.675])
            images = images[..., [2, 1, 0]]
            images = torch.clamp(images, 0, 255)
            
            images = images.cpu().numpy().astype(np.uint8)
            
            total_width = images.shape[2] * 3
            total_height = images.shape[1] * 2
            new_image = Image.new('RGB', (total_width, total_height))
            x_offset = 0
            y_offset = 0
            for counter, camera_id in enumerate([2, 0, 1, 4, 3 ,5]):
                img = Image.fromarray(images[camera_id])
                # Flip Back cameras horizontally, for better visualization
                if camera_id in [4, 3, 5]:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                new_image.paste(img, (x_offset, y_offset))
                x_offset += img.width
                
                if (counter + 1) % 3 == 0:
                    x_offset = 0
                    y_offset += img.height
            
            # 960 width
            new_image = new_image.resize((960, int(960 * new_image.height / new_image.width)))
            
            feature = model.module.pts_bbox_head.feature[feature_scale_lvl]
            logit = model.module.logits[feature_scale_lvl].permute(0, 2, 3, 4, 1)

            scale_ratio = 2**(len(model.module.logits) - 1 - feature_scale_lvl)
            label = multiscale_supervision(data['gt_occ'].clone(), scale_ratio, feature.shape)

            feature = torch.flatten(feature, 0, 3).cpu()
            label = torch.flatten(label).cpu()
            logit = torch.flatten(logit, 0, 3).cpu()
            
            log_prob = gmm.log_prob(feature[:, None, :])
            log_prob = torch.clamp(log_prob, min=-100000, max=100000)
            log_prob = log_prob + prior_log_prob
            
            gmm_uncertainty = torch.logsumexp(log_prob, dim=-1)
            gmm_uncertainty = gmm_uncertainty.view(1, wl, wl, h).cpu().numpy()
            
            softmax = torch.softmax(logit, dim=-1)
            pred_cls_ids = torch.max(softmax, dim=-1)[1]
            pred_cls_ids = pred_cls_ids.view(1, wl, wl, h).cpu().numpy()

            np.save(f"{save_dir}/gmm_log_density{str(i).zfill(4)}.npy", gmm_uncertainty)
            np.save(f"{save_dir}/gmm_predictions{str(i).zfill(4)}.npy", pred_cls_ids)
            new_image.save(f"{save_dir}/cameras_{str(i).zfill(4)}.jpg")

            if i == num_samples - 1:
                break


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
        raise NotImplementedError("Only support single GPU evaluation")

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

    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    num_samples = len(data_loader)
    num_classes = 1 + len(dataset.class_names)
    device = torch.device('cpu')
    feature_scale_lvl = args.feature_scale_lvl
    
    gmm = torch.load(
        os.path.join(os.path.dirname(args.checkpoint), f'train_gmm_scale_{feature_scale_lvl}.pt'),
        map_location=device
    )
    prior_log_prob = torch.load(
        os.path.join(os.path.dirname(args.checkpoint), f'train_prior_log_prob_scale_{feature_scale_lvl}.pt'),
        map_location=device
    )

    save_dir = "clean" if args.overwrite_nuscenes_root is None else "_".join(args.overwrite_nuscenes_root.split("/")[-2:])
    save_dir += f"_scale{feature_scale_lvl}"
    save_dir = os.path.join('/workspace/work_dirs/video', save_dir)
    os.makedirs(save_dir, exist_ok=True)

    gmm_video_v3(
        model=model,
        gmm=gmm,
        prior_log_prob=prior_log_prob,
        data_loader=data_loader,
        feature_scale_lvl=feature_scale_lvl,
        num_samples=num_samples,
        save_dir=save_dir
    )


if __name__ == '__main__':
    main()
