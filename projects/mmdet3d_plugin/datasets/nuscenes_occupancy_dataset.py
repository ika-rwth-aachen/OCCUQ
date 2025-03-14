import copy

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random
import pdb, os


@DATASETS.register_module()
class CustomNuScenesOccDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self,
                 occ_size,
                 pc_range,
                 use_semantic=False,
                 classes=None,
                 overlap_test=False,
                 overwrite_nuscenes_root=None,
                 scenes_to_remove_filepath=None,
                 overwrite_nuscenes_root_for_cameras=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"],
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.overlap_test = overlap_test
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.use_semantic = use_semantic
        self.class_names = classes
        self.overwrite_nuscenes_root = overwrite_nuscenes_root
        self.overwrite_nuscenes_root_for_cameras = overwrite_nuscenes_root_for_cameras
        self.scenes_to_remove_filepath = scenes_to_remove_filepath
        
        if self.overwrite_nuscenes_root is not None:
            print(f"Overwriting nuscenes root for cameras: {self.overwrite_nuscenes_root_for_cameras}")
            print(f"New nuscenes root: {self.overwrite_nuscenes_root}")
        else:
            print("Not overwriting nuscenes root for cameras")
        
        if self.scenes_to_remove_filepath is not None:
            self.remove_samples()

        self._set_group_flag()
    
    def remove_samples(self):
        with open(self.scenes_to_remove_filepath, 'r') as f:
            lines = f.readlines()
        self.scenes_to_remove = [line.strip() for line in lines]
        
        len_data_infos_before = len(self.data_infos)
        
        # remove samples from data_infos
        self.data_infos = [info for info in self.data_infos if os.path.basename(info['lidar_path']) not in self.scenes_to_remove]
        
        # compute fraction of data samples remaining
        len_data_infos_after = len(self.data_infos)
        fraction_remaining = len_data_infos_after / len_data_infos_before
        print(f"Removed {len_data_infos_before - len_data_infos_after} samples from dataset. {fraction_remaining:.2f} of data remaining.")
        print(f"Training on {len(self.data_infos)} samples.")

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            occ_path=info['occ_path'],
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range)
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():

                if self.overwrite_nuscenes_root is not None and cam_type in self.overwrite_nuscenes_root_for_cameras:
                    img_path = cam_info['data_path'].split("nuscenes")[1][1:]
                    img_path = os.path.join(self.overwrite_nuscenes_root, img_path)
                else:
                    img_path = cam_info['data_path']
                
                image_paths.append(img_path)
                # obtain lidar to image transformation matrix

                if 'lidar2cam' in cam_info.keys():
                    lidar2cam_rt = cam_info['lidar2cam'].T
                else:
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

                

                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            info = self.data_infos[idx]
            
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        return results, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        results, tmp_dir = self.format_results(results, jsonfile_prefix)
        results_dict = {}
        if self.use_semantic:
            class_names = {0: 'IoU'}
            class_num = len(self.class_names) + 1
            for i, name in enumerate(self.class_names):
                class_names[i + 1] = self.class_names[i]
            
            results = np.stack(results, axis=0).mean(0)
            mean_ious = []
            
            for i in range(class_num):
                tp = results[i, 0] #TP
                p = results[i, 1] #TP+FN
                g = results[i, 2] #TP+FP
                union = p + g - tp #TP+FN+FP
                mean_ious.append(tp / union) #TP/(TP+FN+FP)
            
            for i in range(class_num):
                results_dict[class_names[i]] = mean_ious[i]
            results_dict['mIoU'] = np.mean(np.array(mean_ious)[1:])

            results_dict['Avg_ECE_per_Scene'] = results[class_num][0]
            results_dict['Avg_NLL_per_Scene'] = results[class_num+1][0]
            results_dict['Avg_AUROC_per_Scene_MI'] = results[class_num+2][0]
            results_dict['Avg_AUROC_per_Scene_PE'] = results[class_num+3][0]
            results_dict['Avg_FPR95_per_Scene_MI'] = results[class_num+4][0]
            results_dict['Avg_FPR95_per_Scene_PE'] = results[class_num+5][0]

        else:
            results = np.stack(results, axis=0).mean(0)
            results_dict={'Acc':results[0],
                          'Comp':results[1],
                          'CD':results[2],
                          'Prec':results[3],
                          'Recall':results[4],
                          'F-score':results[5]}

        return results_dict
