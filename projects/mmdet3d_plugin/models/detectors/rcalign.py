# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations

from mmcv.ops import Voxelization
from mmdet3d.models import builder
from torch.nn import functional as F


@DETECTORS.register_module()
class RCAlign(MVXTwoStageDetector):
    """RCAlign."""

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
                 num_frame_head_grads=2,
                 num_frame_backbone_grads=2,
                 num_frame_losses=2,
                 stride=[16],
                 position_level=[0],
                 aux_2d_only=True,
                 single_test=False,
                 pretrained=None,
                 radar_voxel_layer=None,
                 radar_voxel_encoder=None,
                 radar_middle_encoder=None,
                 radar_roi_head=None,
                 out_voxel_layer=None,
                 out_voxel_encoder=None,
                 out_middle_encoder=None,
                 radar_enhance_fusion=None,
                 ):
        super(RCAlign, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                                        pts_middle_encoder, pts_fusion_layer,
                                        img_backbone, pts_backbone, img_neck, pts_neck,
                                        pts_bbox_head, img_roi_head, img_rpn_head,
                                        train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.single_test = single_test
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.aux_radar_only = False
        self.test_flag = False

        if radar_voxel_layer:
            self.radar_voxel_layer = Voxelization(**radar_voxel_layer)
        if radar_voxel_encoder:
            self.radar_voxel_encoder = builder.build_voxel_encoder(
                radar_voxel_encoder)
        if radar_middle_encoder:
            self.radar_middle_encoder = builder.build_middle_encoder(
                radar_middle_encoder)
        if radar_roi_head:
            self.radar_roi_head = builder.build_head(
                radar_roi_head)

        if out_voxel_layer:
            self.out_voxel_layer = Voxelization(**out_voxel_layer)
        if out_voxel_encoder:
            self.out_voxel_encoder = builder.build_voxel_encoder(
                out_voxel_encoder)
        if out_middle_encoder:
            self.out_middle_encoder = builder.build_middle_encoder(
                out_middle_encoder)
        if radar_enhance_fusion:
            self.radar_enhance_fusion = builder.build_head(radar_enhance_fusion)
        # self.fuseFPN = torch.nn.Conv2d(in_channels=64*3, out_channels=64, kernel_size=1, stride=1)

    @torch.no_grad()
    def radar_voxelize(self, points, second=False):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []

        for res in points:
            if second:
                res_voxels, res_coors, res_num_points = self.out_voxel_layer(res)
            else:
                res_voxels, res_coors, res_num_points = self.radar_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
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

        if self.training or training_mode:
            for i in self.position_level:
                BN, C, H, W = img_feats[i].size()
                img_feat_reshaped = img_feats[i].view(B, len_queue, int(BN / B / len_queue), C, H, W)
                img_feats_reshaped.append(img_feat_reshaped)
        else:
            for i in self.position_level:
                BN, C, H, W = img_feats[i].size()
                img_feat_reshaped = img_feats[i].view(B, int(BN / B / len_queue), C, H, W)
                img_feats_reshaped.append(img_feat_reshaped)

        return img_feats_reshaped

    def extract_radar_feat(self, radar):
        """Extract features of points."""
        voxels, num_points, coors = self.radar_voxelize(radar)
        voxel_features = self.radar_voxel_encoder(voxels, num_points, coors)
        batch_size = torch.as_tensor(coors[-1, 0] + 1, dtype=torch.int)
        x = self.radar_middle_encoder(voxel_features, coors, batch_size)
        return x

    def out_to_bev_mask(self, out):
        voxels, num_points, coors = self.radar_voxelize(out, True)
        voxel_features = self.out_voxel_encoder(voxels, num_points, coors)
        voxel_features = voxel_features.unsqueeze(1)
        batch_size = torch.as_tensor(coors[-1, 0] + 1, dtype=torch.int)
        x = self.out_middle_encoder(voxel_features, coors, batch_size)

        return x

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, T, radar, training_mode=False):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, T, training_mode)
        radar_feats = self.extract_radar_feat(radar)
        return img_feats, radar_feats

    def obtain_history_memory(self,
                              gt_bboxes_3d=None,
                              gt_labels_3d=None,
                              gt_bboxes=None,
                              gt_labels=None,
                              img_metas=None,
                              centers2d=None,
                              depths=None,
                              gt_bboxes_ignore=None,
                              **data):
        losses = dict()
        T = data['img'].size(1)
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses
        for i in range(T):
            requires_grad = False
            return_losses = False
            data_t = dict()
            for key in data:
                if key == 'img_feats':
                    data_t[key] = [feat[:, i] for feat in data[key]]
                elif key in['radar_feats', 'radar_reference_points']:
                    data_t[key] = data[key]
                else:
                    data_t[key] = data[key][:, i]

            data_t['img_feats'] = data_t['img_feats']
            data_t['radar_feats'] = data_t['radar_feats']
            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True
            loss = self.forward_pts_train(gt_bboxes_3d[i],
                                          gt_labels_3d[i], gt_bboxes[i],
                                          gt_labels[i], img_metas[i], centers2d[i], depths[i],
                                          requires_grad=requires_grad, return_losses=return_losses, **data_t)
            if loss is not None:
                for key, value in loss.items():
                    losses['frame_' + str(i) + "_" + key] = value
        return losses

    def prepare_radar_location(self, radar_feat, voxel_size, point_cloud_range):
        range_x = [point_cloud_range[0], point_cloud_range[3]]
        range_y = [point_cloud_range[1], point_cloud_range[4]]
        range_z = [point_cloud_range[2], point_cloud_range[5]]

        x_stride = voxel_size[0]
        y_stride = voxel_size[1]

        bs, _, h, w = radar_feat.size()
        device = radar_feat.device
        # norm
        shifts_x = (torch.arange(
            range_x[0], range_x[1], step=x_stride,
            dtype=torch.float32, device=device
        ) + x_stride / 2 - range_x[0]) / (range_x[1]-range_x[0])
        shifts_y = (torch.arange(
            range_y[0], range_y[1], step=y_stride,
            dtype=torch.float32, device=device
        ) + y_stride / 2 - range_y[0]) / (range_y[1]-range_y[0])
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        shift_z = (1.-range_z[0]) / (range_z[1]-range_z[0])
        shift_z = shift_x.new_full(shift_x.size(), shift_z, dtype=shift_x.dtype, device=device)

        locations = torch.stack((shift_x, shift_y, shift_z), dim=1)

        locations = locations.reshape(h, w, 3)

        return locations[None].repeat(bs, 1, 1, 1)

    @property
    def with_radar_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, 'radar_roi_head') and self.radar_roi_head is not None

    def forward_roi_head(self, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes': None}
        else:
            outs_roi = self.img_roi_head(**data)
            return outs_roi

    def forward_radar_roi_head(self,location, radar_feat):
        if (self.aux_radar_only and not self.training) or not self.with_radar_roi_head:
            return {'topk_indexes': None}
        else:
            outs_roi = self.radar_roi_head(location, radar_feat)
            return outs_roi

    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        location = self.prepare_radar_location(data['radar_feats'], self.radar_voxel_layer.voxel_size, self.radar_voxel_layer.point_cloud_range)
        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs_radar_roi = self.forward_radar_roi_head(location, **data)
                topk_indexes = outs_radar_roi['topk_indexes']
                outs = self.pts_bbox_head(img_metas, topk_indexes, **data)
            self.train()

        else:
            outs_roi = self.forward_roi_head(**data)
            outs_radar_roi = self.forward_radar_roi_head(location, data['radar_feats'])
            topk_indexes = outs_radar_roi['topk_indexes']
            pre_location = outs_radar_roi['pred_bev_centers3d']
            outs = self.pts_bbox_head(img_metas, topk_indexes, pre_location, **data)
            out_bev_mask = self.out_to_bev_mask(outs['all_bbox_preds'][-1][..., :3])
            radar_enhacne_feature = self.radar_enhance_fusion(data['radar_feats'], out_bev_mask)
            outs_sec_radar_roi = self.forward_radar_roi_head(location, radar_enhacne_feature)
        if return_losses:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)
            if self.with_img_roi_head:
                loss2d_inputs = [gt_bboxes, gt_labels, centers2d, outs_roi, depths, img_metas]
                losses2d = self.img_roi_head.loss(*loss2d_inputs)
                losses.update(losses2d)
            if self.with_radar_roi_head:
                loss_radar_input = [gt_bboxes_3d, gt_labels_3d, outs_radar_roi, img_metas]
                losses_radar = self.radar_roi_head.loss(*loss_radar_input)
                losses.update(losses_radar)

                loss_second_radar_input = [gt_bboxes_3d, gt_labels_3d, outs_sec_radar_roi, img_metas, True, data['radar_feats'], radar_enhacne_feature]
                losses_sec_radar = self.radar_roi_head.loss(*loss_second_radar_input)
                losses.update(losses_sec_radar)
            return losses
        else:
            return None

    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
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
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas', 'radar']:
                data[key] = list(zip(*data[key]))
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      radar=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.extract_feat
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.test_flag:  # for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False

        T = data['img'].size(1)

        prev_img = data['img'][:, :-self.num_frame_backbone_grads]
        rec_img = data['img'][:, -self.num_frame_backbone_grads:]

        rec_radar = radar[0]

        rec_img_feats, rec_radar_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads, rec_radar)

        if T - self.num_frame_backbone_grads > 0:
            self.eval()
            with torch.no_grad():
                prev_img_feats = self.extract_feat(prev_img, T - self.num_frame_backbone_grads, True)
            self.train()
            data['img_feats'] = [torch.cat([prev_img_feats[i], rec_img_feats[i]], dim=1) for i in
                                 range(len(self.position_level))]
            data['radar_feats'] = rec_radar_feats
        else:
            data['img_feats'] = rec_img_feats
            data['radar_feats'] = rec_radar_feats

        losses = self.obtain_history_memory(gt_bboxes_3d,
                                            gt_labels_3d, gt_bboxes,
                                            gt_labels, img_metas, centers2d, depths, gt_bboxes_ignore, **data)

        return losses

    def forward_test(self, img_metas, rescale, **data):
        self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key != 'img':
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        outs_roi = self.forward_roi_head(**data)
        location = self.prepare_radar_location(data['radar_feats'], self.radar_voxel_layer.voxel_size,
                                               self.radar_voxel_layer.point_cloud_range)
        outs_radar_roi = self.forward_radar_roi_head(location, data['radar_feats'])
        topk_indexes = outs_radar_roi['topk_indexes']
        pre_location = outs_radar_roi['pred_bev_centers3d']
        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = data['img'].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = data['img'].new_ones(1)

        outs = self.pts_bbox_head(img_metas, topk_indexes, pre_location, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        # radar_reference_point = []
        # num_radars_to_select = 30
        # for rrp_idx in range(len(data['radar'])):
        #     selected_radars_indices = torch.randint(0, data['radar'][rrp_idx].size(0), (num_radars_to_select,))
        #     selected_radars = data['radar'][rrp_idx][selected_radars_indices, :3]
        #     radar_reference_point.append(selected_radars)
        # data['radar_reference_points'] = torch.stack(radar_reference_point)

        data['img_feats'] = self.extract_img_feat(data['img'], 1)
        data['radar_feats'] = self.extract_radar_feat([data['radar'][0]]).to(torch.float32)

        # from tools.utils.feature_visualization import Visualization
        # vis = Visualization()
        # vis.showFeature2Img(img_metas[0]['filename'], data['img_feats'][0])

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

