# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean, bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh)
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from projects.mmdet3d_plugin.models.utils.misc import draw_heatmap_gaussian, apply_radar_center_offset, gaussian_radius
from mmdet.core import bbox_overlaps
from mmdet3d.models.utils import clip_sigmoid
import random


@HEADS.register_module()
class RadarHead(nn.Module):
    def __init__(self,
                 in_channels=256,
                 embed_dims=256,
                 loss_bev_centerness=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bev_centers3d=dict(type='L1Loss', loss_weight=5.0),
                 loss_kd=dict(type='L1Loss', loss_weight=5.0),
                 train_ratio=1.0,
                 infer_ratio=1.0,
                 use_hybrid_tokens=False,
                 radar_voxel_size=[0.8, 0.8, 8],
                 point_cloud_range=[-51.2, -51.2, -3, 51.2, 51.2, 5],
                 train_cfg=dict(
                     min_overlap=0.1,
                     out_size_factor=4,
                     assigner2d=dict(
                         type='HungarianAssignerRadar',
                         centers2d_cost=dict(type='BBox3DL1Cost', weight=1.0))),
                 test_cfg=dict(max_per_img=100),
                 ):

        if train_cfg:
            assert 'assigner2d' in train_cfg, 'assigner2d should be provided '\
                'when train_cfg is set.'
            assigner2d = train_cfg['assigner2d']

            self.assigner2d = build_assigner(assigner2d)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.in_channels = in_channels
        self.embed_dims = embed_dims

        self.train_ratio = train_ratio
        self.infer_ratio = infer_ratio

        self.radar_voxel_size = radar_voxel_size
        self.point_cloud_range = point_cloud_range

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.use_hybrid_tokens = use_hybrid_tokens

        super().__init__()

        self.loss_bev_centers3d = build_loss(loss_bev_centers3d)
        self.loss_bev_centerness = build_loss(loss_bev_centerness)
        self.loss_kd = build_loss(loss_kd)

        self._init_layers()

    def _init_layers(self):
        self.bev_reg = nn.Sequential(
            nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=self.embed_dims),
            nn.ReLU(),)

        self.bev_cls = nn.Sequential(
            nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=self.embed_dims),
            nn.ReLU(),)

        self.bev_centerness = nn.Conv2d(self.embed_dims, 1, kernel_size=1)
        self.bev_center3d = nn.Conv2d(self.embed_dims, 3, kernel_size=1)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.bev_centerness.bias, bias_init)

    def forward(self, location, radar_feat):
        x = radar_feat
        bs, c, h, w = x.shape
        num_tokens = h * w

        # focal sampling
        if self.training:
            if self.use_hybrid_tokens:
                sample_ratio = random.uniform(0.2, 1.0)
            else:
                sample_ratio = self.train_ratio
            num_sample_tokens = int(num_tokens * sample_ratio)

        else:
            sample_ratio = self.infer_ratio
            num_sample_tokens = int(num_tokens * sample_ratio)

        cls_feat = self.bev_cls(x)
        bev_centerness = self.bev_centerness(cls_feat)
        bev_centerness = bev_centerness.permute(0, 2, 3, 1).reshape(bs, -1, 1)

        reg_feat = self.bev_reg(x)
        bev_centers3d_offset = self.bev_center3d(reg_feat).permute(0, 2, 3, 1).contiguous()
        # bev_centers2d = self.apply_radar_center_offset(location, bev_centers2d_offset)
        bev_centers3d = self.apply_radar_center_offset(location, bev_centers3d_offset)
        pred_bev_centers3d = bev_centers3d.view(bs, -1, 3)

        # from tools.utils.feature_visualization import Visualization
        # vis = Visualization()
        # vis.plot_bev_point(pred_bev_centers3d)

        sample_weight = bev_centerness.detach().view(bs, -1, 1).sigmoid()

        score, topk_indexes = torch.topk(sample_weight, num_sample_tokens, dim=1)

        outs = {
            'pred_bev_centers3d': pred_bev_centers3d,
            'bev_centerness': bev_centerness,
            'topk_indexes': topk_indexes,
            'score': score,
        }

        return outs

    def apply_radar_center_offset(self, locations, center_offset):
        centers_3d = torch.zeros_like(center_offset)
        pc_range = self.point_cloud_range

        #Denorm
        locations[..., 0] = locations[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        locations[..., 1] = locations[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        locations[..., 2] = locations[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

        centers_3d[..., 0] = locations[..., 0] + center_offset[..., 0]  # x
        centers_3d[..., 1] = locations[..., 1] + center_offset[..., 1]  # y
        centers_3d[..., 2] = locations[..., 2] + center_offset[..., 2]  # z

        #norm
        centers_3d[..., 0] = (locations[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
        centers_3d[..., 1] = (locations[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
        centers_3d[..., 2] = (locations[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])
        return centers_3d

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes3d_list,
             gt_labels3d_list,
             preds_dicts,
             img_metas,
             kd=False,
             ori_radar_feat=None,
             sec_radar_feat=None,
             gt_bboxes_ignore=None):

        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        #test = [torch.cat((i.gravity_center, i.tensor[:, 3:]),dim=1) for i in gt_bboxes3d_list ]
        pred_bev_centers3d = preds_dicts['pred_bev_centers3d']
        bev_centerness = preds_dicts['bev_centerness']

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        device = pred_bev_centers3d.device
        all_gt_bboxes3d_list = [torch.cat((bboxes3D.gravity_center, bboxes3D.tensor[:, 3:]), dim=1).to(device) for bboxes3D in gt_bboxes3d_list]
        all_gt_labels3d_list = gt_labels3d_list
        all_centers3d_list = [center3d.gravity_center.to(device) for center3d in gt_bboxes3d_list]

        bev_centers3d_losses, bev_centerness_losses, kd_loss = \
            self.loss_single(pred_bev_centers3d, bev_centerness, all_gt_bboxes3d_list, all_gt_labels3d_list, all_centers3d_list, img_metas,
                             kd, ori_radar_feat, sec_radar_feat, gt_bboxes_ignore)

        if kd:
            loss_dict['sec_bev_centers3d_losses'] = bev_centers3d_losses * 0.1
            loss_dict['sec_bev_centerness_losses'] = bev_centerness_losses
            loss_dict['kd_losses'] = kd_loss
        else:
            loss_dict['bev_centers3d_losses'] = bev_centers3d_losses
            loss_dict['bev_centerness_losses'] = bev_centerness_losses

        return loss_dict

    def loss_single(self,
                    pred_bev_centers3d,
                    bev_centerness,
                    gt_bboxes3d_list,
                    gt_labels3d_list,
                    all_centers3d_list,
                    img_metas,
                    kd=False,
                    ori_radar_feat=None,
                    sec_radar_feat=None,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """

        num_bevmaps = pred_bev_centers3d.size(0)
        centers3d_preds_list = [pred_bev_centers3d[i] for i in range(num_bevmaps)]

        cls_reg_targets = self.get_targets(centers3d_preds_list,
                                           gt_bboxes3d_list, gt_labels3d_list, all_centers3d_list,
                                           img_metas, gt_bboxes_ignore_list)
        (centers3d_targets_list, centers3d_weight, num_total_pos, num_total_neg) = cls_reg_targets

        centers3d_targets_list = torch.cat(centers3d_targets_list, 0)
        centers3d_weight = torch.cat(centers3d_weight, 0)

        # img_shape = [img_metas[0]['pad_shape'][0]] * num_bevmaps
        bev_shape = [(128, 128, 1)] * num_bevmaps
        (heatmaps,) = multi_apply(self._get_bev_heatmap_single, all_centers3d_list, gt_bboxes3d_list, bev_shape)

        heatmaps = torch.stack(heatmaps, dim=0)
        centerness = clip_sigmoid(bev_centerness)
        loss_bev_centerness = self.loss_bev_centerness(
            centerness,
            heatmaps.view(num_bevmaps, -1, 1),
            avg_factor=max(num_total_pos, 1))

        # regression L1 loss
        pred_bev_centers3d = pred_bev_centers3d.view(-1, 3)
        loss_bev_centers3d = self.loss_bev_centers3d(
            pred_bev_centers3d, centers3d_targets_list, centers3d_weight, avg_factor=num_total_pos)

        #KD loss
        loss_kd = None
        if kd:
            loss_kd = self.loss_kd(ori_radar_feat, sec_radar_feat)

        return loss_bev_centers3d, loss_bev_centerness, loss_kd

    def _get_heatmap_single(self, obj_centers2d, obj_bboxes, img_shape):
        img_h, img_w, _ = img_shape
        heatmap = torch.zeros(img_h // self.stride, img_w // self.stride, device=obj_centers2d.device)
        if len(obj_centers2d) != 0:
            l = obj_centers2d[..., 0:1] - obj_bboxes[..., 0:1]
            t = obj_centers2d[..., 1:2] - obj_bboxes[..., 1:2]
            r = obj_bboxes[..., 2:3] - obj_centers2d[..., 0:1]
            b = obj_bboxes[..., 3:4] - obj_centers2d[..., 1:2]
            bound = torch.cat([l, t, r, b], dim=-1)
            radius = torch.ceil(torch.min(bound, dim=-1)[0] / 16)
            radius = torch.clamp(radius, 1.0).cpu().numpy().tolist()
            for center, r in zip(obj_centers2d, radius):
                heatmap = draw_heatmap_gaussian(heatmap, center / 16, radius=int(r), k=1)
        return (heatmap,)

    def _get_bev_heatmap_single(self, obj_centers3d, obj_bboxes3d, bev_shape):
        heatmap = obj_bboxes3d.new_zeros((bev_shape[1], bev_shape[0]))
        width = obj_bboxes3d[:, 3]
        length = obj_bboxes3d[:, 4]

        pc_range = obj_bboxes3d.new_tensor(self.point_cloud_range)

        width = width / self.radar_voxel_size[0] / self.train_cfg['out_size_factor']
        length = length / self.radar_voxel_size[1] / self.train_cfg['out_size_factor']

        radius = gaussian_radius((length, width), min_overlap=self.train_cfg['min_overlap'])
        radius = torch.clamp(radius, 1.0).cpu().numpy().tolist()

        x = obj_centers3d[:, 0]
        y = obj_centers3d[:, 1]

        coor_x = (x - pc_range[0]) / self.radar_voxel_size[0] / self.train_cfg['out_size_factor']
        coor_y = (y - pc_range[1]) / self.radar_voxel_size[1] / self.train_cfg['out_size_factor']

        center = torch.cat([coor_x.unsqueeze(1), coor_y.unsqueeze(1)], dim=-1)
        center_int = center.to(torch.int32)
        for center, r in zip(center_int, radius):
            heatmap = draw_heatmap_gaussian(heatmap, center, radius=int(r), k=1)
        return (heatmap,)

    def get_targets(self,
                    centers3d_preds_list,
                    gt_bboxes3d_list,
                    gt_labels3d_list,
                    all_centers3d_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_bevmaps = len(centers3d_preds_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_bevmaps)
        ]
        img_meta = {'pad_shape': img_metas[0]['pad_shape'][0]}
        img_meta_list = [img_meta for _ in range(num_bevmaps)]
        # print(1)
        (centers3d_targets_list, centers3d_weight, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, centers3d_preds_list,
            gt_bboxes3d_list, gt_labels3d_list, all_centers3d_list,
            img_meta_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (centers3d_targets_list, centers3d_weight, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           pred_centers3d,
                           gt_bboxes3d,
                           gt_labels3d,
                           centers3d,
                           img_meta,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indexes for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indexes for each image.
                - neg_inds (Tensor): Sampled negative indexes for each image.
        """

        num_bboxes = pred_centers3d.size(0)
        # assigner and sampler
        assign_result = self.assigner2d.assign(pred_centers3d, gt_bboxes3d,
                                               gt_labels3d, centers3d, img_meta, gt_bboxes_ignore)
        # sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        sampling_result = self.sampler.sample(assign_result, pred_centers3d, gt_bboxes3d)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        factor = pred_centers3d.new_tensor(self.point_cloud_range)

        # centers3d target
        centers3d_weights = torch.zeros_like(pred_centers3d)
        centers3d_weights[pos_inds] = 1.0
        centers3d_targets = pred_centers3d.new_full((num_bboxes, 3), 0.0, dtype=torch.float32)
        if gt_bboxes3d.numel() == 0:
            # hack for index error case
            assert sampling_result.pos_assigned_gt_inds.numel() == 0
            centers3d_labels = torch.empty_like(gt_bboxes3d).view(-1, 3)
        else:
            centers3d_labels = centers3d[sampling_result.pos_assigned_gt_inds.long(), :]
        centers3d_labels_normalized = (centers3d_labels - factor[0:3]) / (factor[3:6] - factor[0:3])
        centers3d_targets[pos_inds] = centers3d_labels_normalized
        return (centers3d_targets, centers3d_weights, pos_inds, neg_inds)
