# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Modified by Institute for Automotive Engineering of RWTH Aachen University
#  Modified by Computer Vision Group of RWTH Aachen University
#  by Severin Heidrich, Till Beemelmanns, Alexey Nekrasov
# ---------------------------------------------

#import open3d as o3d
from tkinter.messagebox import NO
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.datasets.evaluation_metrics import evaluation_reconstruction, evaluation_semantic
from sklearn.metrics import confusion_matrix as CM
import time, yaml, os
import torch.nn as nn
import csv

def entropy_prob(probs):
    logp = torch.log(probs + 1e-12)
    plogp = probs * logp
    entropy = -torch.sum(plogp, axis=1)
    return entropy

def mutual_information_prob(probs, predictive_entropy):
    logp = torch.log(probs + 1e-12)
    plogp = probs * logp
    exp_entropies = torch.mean(-torch.sum(plogp, axis=1), axis=0)
    mi = predictive_entropy - exp_entropies
    return mi



@DETECTORS.register_module()
class SurroundOcc(MVXTwoStageDetector):
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_semantic=True,
                 is_vis=False,
                 store_dir=None
                 ):

        super(SurroundOcc,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        self.use_semantic = use_semantic
        self.is_vis = is_vis

        self.pred = None
        self.store_dir = store_dir

        # for deep ensembles
        # model_pathes = [
        #     "/workspace/work_dirs/surroundocc_seed_1/epoch_24.pth",
        #     "/workspace/work_dirs/surroundocc_seed_2/epoch_24.pth",
        #     "/workspace/work_dirs/surroundocc_seed_3/epoch_24.pth",
        #     "/workspace/work_dirs/surroundocc_seed_4/epoch_24.pth",
        #     "/workspace/work_dirs/surroundocc_seed_5/epoch_24.pth",
        # ]
        
        # self.state_dicts = []
        # for model_path in model_pathes:
        #     self.state_dicts.append(torch.load(model_path, map_location='cuda')['state_dict'])


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          gt_occ,
                          img_metas):

        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_occ, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, mcdropout=False, deepensembles=False, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        elif mcdropout:
            return self.forward_mcd_test(**kwargs)
        elif deepensembles:
            return self.forward_de_test(**kwargs)
        else:
            return self.forward_test(**kwargs)
    

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img_metas=None,
                      gt_occ=None,
                      img=None
                      ):

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_occ,
                                             img_metas)

        losses.update(losses_pts)
        return losses


    def forward_test(self, img_metas, img=None, gt_occ=None, **kwargs):
        output = self.simple_test(img_metas, img, **kwargs)
        
        pred_occ = output['occ_preds']
        self.logits = pred_occ
        if type(pred_occ) == list:
            pred_occ = pred_occ[-1]
        
        
        self.pred = torch.softmax(pred_occ, dim=1)
        confidences, pred_cls_id = torch.max(torch.softmax(pred_occ, dim=1), dim=1)
        class_num = pred_occ.shape[1]
        
        if self.is_vis:
            return pred_occ.shape[0]
        
        eval_results = evaluation_semantic(pred_cls_id, confidences, gt_occ, img_metas[0], class_num, self.pred)

        return {'evaluation': eval_results}


    def forward_mcd_test(self, img_metas, img=None, gt_occ=None, **kwargs):
        n_fw_passes = 5
        
        probs_list = []
        for _ in range(n_fw_passes):
            tmp_img = copy.deepcopy(img)
            tmp_img_metas = copy.deepcopy(img_metas)
            pred_occ = self.simple_test(tmp_img_metas, tmp_img, **kwargs)['occ_preds']
            pred_occ = pred_occ[-1]
            probs = torch.softmax(pred_occ, dim=1)
            probs_list.append(probs.squeeze())
        
        # stack output
        probs = torch.stack(probs_list, dim=0)
        self.pred = torch.mean(probs, axis=0).unsqueeze(0)
        confidences, pred_occ = torch.max(self.pred, dim=1)
        class_num = self.pred.shape[1]

        predictive_entropy = entropy_prob(self.pred)
        mut_info = mutual_information_prob(probs, predictive_entropy)

        if self.is_vis:
            self.save_value_to_csv(mut_info.mean().cpu().numpy(), filename='mcd_mut_info_per_sample.csv')
            self.save_value_to_csv(predictive_entropy.mean().cpu().numpy(), filename='mcd_predictive_entropy_per_sample.csv')
            return pred_occ.shape[0]
        
        eval_results = evaluation_semantic(pred_occ, confidences, gt_occ, img_metas[0], class_num, softmax=self.pred)

        return {'evaluation': eval_results}



    def forward_de_test(self, img_metas, img=None, gt_occ=None, **kwargs):
        logits_list = []
        probs_list = []
        for state_dict in self.state_dicts:
            self.load_state_dict(state_dict, strict=False)
            tmp_img = copy.deepcopy(img)
            tmp_img_metas = copy.deepcopy(img_metas)
            logits_occ = self.simple_test(tmp_img_metas, tmp_img, **kwargs)['occ_preds']
            logits_occ = logits_occ[-1]

            probs = torch.softmax(logits_occ, dim=1)
            
            probs_list.append(probs.squeeze())
        
        # stack output
        probs = torch.stack(probs_list, dim=0)
        self.pred = torch.mean(probs, axis=0).unsqueeze(0)
        confidences, pred_occ = torch.max(self.pred, dim=1)
        class_num = self.pred.shape[1]

        predictive_entropy = entropy_prob(self.pred)
        mut_info = mutual_information_prob(probs, predictive_entropy)

        #self.save_value_to_csv(mut_info.mean().cpu().numpy(), filename='de5mi_per_sample.csv')
        #self.save_value_to_csv(predictive_entropy.mean().cpu().numpy(), filename='de5pe_per_sample.csv')

        if self.is_vis:
            self.save_entropy(predictive_entropy, img_metas)
            self.save_softmax_prediction(self.pred, img_metas)
            return pred_occ.shape[0]
        
        eval_results = evaluation_semantic(pred_occ, confidences, gt_occ, img_metas[0], class_num, softmax=self.pred)

        return {'evaluation': eval_results}


    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas)
        return outs

    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        output = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale)

        return output

    def generate_output(self, pred_occ, img_metas):
        import open3d as o3d
        
        color_map = np.array(
                [
                    [0, 0, 0, 255],
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
                    [139, 137, 137, 255],
                    [75, 0, 75, 255],  # sidewalk             dard purple
                    [150, 240, 80, 255],  # terrain              light green
                    [230, 230, 250, 255],  # manmade              white
                    [0, 175, 0, 255],  # vegetation           green
                ]
            )
        
        # pred_occ.shape = torch.Size([1, 17, 200, 200, 16]) because occ_size = [200, 200, 16] and num_classes=17

        if self.use_semantic:
            # produce softmax predictions for each voxel
            _, voxel = torch.max(torch.softmax(pred_occ, dim=1), dim=1)
            # voxel.shape = torch.Size([1, 200, 200, 16]) because not value per class anymore but just one class per voxel            
        else:
            voxel = torch.sigmoid(pred_occ[:, 0])      
        
        for i in range(voxel.shape[0]):
            x = torch.linspace(0, voxel[i].shape[0] - 1, voxel[i].shape[0]) #x.shape = torch.Size([200])
            y = torch.linspace(0, voxel[i].shape[1] - 1, voxel[i].shape[1]) #y.shape = torch.Size([200])
            z = torch.linspace(0, voxel[i].shape[2] - 1, voxel[i].shape[2]) #z.shape = torch.Size([16])
            X, Y, Z = torch.meshgrid(x, y, z) #X.shape = Y.shape = Z.shape = torch.Size([200, 200, 16])
            vv = torch.stack([X, Y, Z], dim=-1).to(voxel.device) #vv.shape = torch.Size([200, 200, 16, 3])
        
            vertices = vv[voxel[i] > 0.5] #vertices.shape = torch.Size([61753, 3])
            vertices[:, 0] = (vertices[:, 0] + 0.5) * (img_metas[i]['pc_range'][3] - img_metas[i]['pc_range'][0]) /  img_metas[i]['occ_size'][0]  + img_metas[i]['pc_range'][0]
            vertices[:, 1] = (vertices[:, 1] + 0.5) * (img_metas[i]['pc_range'][4] - img_metas[i]['pc_range'][1]) /  img_metas[i]['occ_size'][1]  + img_metas[i]['pc_range'][1]
            vertices[:, 2] = (vertices[:, 2] + 0.5) * (img_metas[i]['pc_range'][5] - img_metas[i]['pc_range'][2]) /  img_metas[i]['occ_size'][2]  + img_metas[i]['pc_range'][2]
            #vertices contains the (x,y,z) coordinates of all voxels with class other than 0

            vertices = vertices.cpu().numpy()
    
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            if self.use_semantic:
                semantics = voxel[i][voxel[i] > 0].cpu().numpy() #semantics.shape = (61753,)
                #semantics contains for each voxel (in vertices) the class of the voxel
                color = color_map[semantics] / 255.0 #color.shape = (61753, 4)
                #color contains the color for each voxel (in vertices)
                pcd.colors = o3d.utility.Vector3dVector(color[..., :3])
                vertices = np.concatenate([vertices, semantics[:, None]], axis=-1)
                #vertices now contains the (x,y,z) coordinates and the class of all voxels with class other than 0
    
            save_dir = os.path.join('visual_dir', img_metas[i]['occ_path'].replace('.npy', '').split('/')[-1])
            os.makedirs(save_dir, exist_ok=True)

            #save softmax probabilities
            save_dir_softmax = os.path.join('mlp_sn_predictions_full/seed_0', img_metas[i]['occ_path'].replace('.npy', '').split('/')[-1])
            os.makedirs(save_dir_softmax, exist_ok=True)
            np.save(os.path.join(save_dir_softmax, 'softmax_distribution.npy'), torch.softmax(pred_occ, dim=1).cpu().numpy().astype(np.float16))

            #o3d.io.write_point_cloud(os.path.join(save_dir, 'pred.ply'), pcd)
            #np.save(os.path.join(save_dir, 'pred.npy'), vertices)
            #for cam_id, cam_path in enumerate(img_metas[i]['filename']):
            #    os.system('cp {} {}/{}.jpg'.format(cam_path, save_dir, cam_id))

    def save_value_to_csv(self, value, filename):
        os.makedirs(self.store_dir, exist_ok=True)
        # append value to csv file, if file does not exists, create it
        with open(os.path.join(self.store_dir, filename), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([value])

    def save_softmax_prediction(self, softmax_distribution, img_metas):
        save_dir_softmax = os.path.join(self.store_dir, img_metas[0]['occ_path'].replace('.npy', '').split('/')[-1])
        os.makedirs(save_dir_softmax, exist_ok=True)
        np.save(os.path.join(save_dir_softmax, 'softmax_distribution.npy'), softmax_distribution.cpu().numpy().astype(np.float16))

    def save_entropy(self, entropy, img_metas):
        save_dir_softmax = os.path.join(self.store_dir, img_metas[0]['occ_path'].replace('.npy', '').split('/')[-1])
        os.makedirs(save_dir_softmax, exist_ok=True)
        np.save(os.path.join(save_dir_softmax, 'entropy.npy'), entropy.cpu().numpy().astype(np.float16))

    def generate_softmax_uncertainty_output(self, pred_occ, img_metas):
        import open3d as o3d
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # Create ScalarMappable objects with (reversed) RdYlGn colormap
        scalar_mappable = cm.ScalarMappable(cmap='RdYlGn')

        # Set the range of the scalar mappable to [0, 1]
        scalar_mappable.set_clim(vmin=0, vmax=1)

        _, voxel = torch.max(torch.softmax(pred_occ, dim=1), dim=1)

        entropy = self.get_confidence_from_entropy(pred_occ, pred_occ.shape[1])
        max_softmax_prob = self.max_softmax_prob(pred_occ)

        for i in range(voxel.shape[0]):
            x = torch.linspace(0, voxel[i].shape[0] - 1, voxel[i].shape[0]) #x.shape = torch.Size([200])
            y = torch.linspace(0, voxel[i].shape[1] - 1, voxel[i].shape[1]) #y.shape = torch.Size([200])
            z = torch.linspace(0, voxel[i].shape[2] - 1, voxel[i].shape[2]) #z.shape = torch.Size([16])
            X, Y, Z = torch.meshgrid(x, y, z) #X.shape = Y.shape = Z.shape = torch.Size([200, 200, 16])
            vv = torch.stack([X, Y, Z], dim=-1).to(voxel.device) #vv.shape = torch.Size([200, 200, 16, 3])
        
            vertices = vv[voxel[i] > 0.5] #vertices.shape = torch.Size([61753, 3])
            vertices[:, 0] = (vertices[:, 0] + 0.5) * (img_metas[i]['pc_range'][3] - img_metas[i]['pc_range'][0]) /  img_metas[i]['occ_size'][0]  + img_metas[i]['pc_range'][0]
            vertices[:, 1] = (vertices[:, 1] + 0.5) * (img_metas[i]['pc_range'][4] - img_metas[i]['pc_range'][1]) /  img_metas[i]['occ_size'][1]  + img_metas[i]['pc_range'][1]
            vertices[:, 2] = (vertices[:, 2] + 0.5) * (img_metas[i]['pc_range'][5] - img_metas[i]['pc_range'][2]) /  img_metas[i]['occ_size'][2]  + img_metas[i]['pc_range'][2]
            #vertices contains the (x,y,z) coordinates of all voxels with class other than 0

            vertices = vertices.cpu().numpy()
        
            entropy_pcd = o3d.geometry.PointCloud()
            entropy_pcd.points = o3d.utility.Vector3dVector(vertices)

            max_softmax_prob_pcd = o3d.geometry.PointCloud()
            max_softmax_prob_pcd.points = o3d.utility.Vector3dVector(vertices)

            entropy_uncertainties = entropy[i][voxel[i] > 0].cpu().numpy()
            #entropy_uncertainties contains for each voxel (in vertices) the uncertainty based on entropy

            max_softmax_prob_uncertainties = max_softmax_prob[i][voxel[i] > 0].cpu().numpy()
            #confidence_uncertainties contains for each voxel (in vertices) the uncertainty based on the maximum softmax probability

            # Get colors for the uncertainties
            max_softmax_prob_colors = scalar_mappable.to_rgba(max_softmax_prob_uncertainties)
            entropy_colors = scalar_mappable.to_rgba(entropy_uncertainties)

            max_softmax_prob_pcd.colors = o3d.utility.Vector3dVector(max_softmax_prob_colors[..., :3])
            entropy_pcd.colors = o3d.utility.Vector3dVector(entropy_colors[..., :3])

            max_softmax_prob_vertices = np.concatenate([vertices, max_softmax_prob_uncertainties[:, None]], axis=-1)
            entropy_vertices = np.concatenate([vertices, entropy_uncertainties[:, None]], axis=-1)
            #entropy_vertices now contains the (x,y,z) coordinates and the uncertainty based on entropy of all voxels with class other than 0

            save_dir = os.path.join('visual_dir', img_metas[i]['occ_path'].replace('.npy', '').split('/')[-1])
            os.makedirs(save_dir, exist_ok=True)

            #o3d.io.write_point_cloud(os.path.join(save_dir, 'max_softmax_prob_uncertainties.ply'), max_softmax_prob_pcd)
            #o3d.io.write_point_cloud(os.path.join(save_dir, 'entropy_uncertainties.ply'), entropy_pcd)

    def get_confidence_from_entropy(self, logits, class_num):
        entropy = self.entropy(logits)
        normalized_entropy = entropy / torch.log(torch.tensor(class_num, dtype=entropy.dtype))
        inverted_normalized_entropy = 1 - normalized_entropy
        return inverted_normalized_entropy

    def entropy(self, logits):
        p = torch.softmax(logits, dim=1)
        logp = torch.log_softmax(logits, dim=1)
        plogp = p * logp
        entropy = -torch.sum(plogp, dim=1)
        return entropy

    def max_softmax_prob(self, logits):
        p = torch.softmax(logits, dim=1)
        confidence, _ = torch.max(p, dim=1)
        return confidence
        




    
    
    
    