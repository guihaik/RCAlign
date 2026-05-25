import torch
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
import matplotlib.pyplot as plt

from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch

class CustomHooks:
    def __init__(self, model):
        self.model = model

        #output param about
        self.num_classes = 10
        self.max = 300
        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.post_center_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
        self.CLASSES = self.model.checkpoint['meta']['CLASSES']
        self.score_thr = 0.3

        #deform param about
        self.num_groups = 8
        self.num_levels =4
        self.im2col_step = 64
        self.num_cams = 6

        #get the model that want to handle
        self.mulheadAttention = self.model.detector.pts_bbox_head.transformer.decoder.layers[-1].attentions[0].attn  # 0 multiheadAttention
        self.DeformableAttention = self.model.detector.pts_bbox_head.transformer.decoder.layers[-1].attentions[1]    # 1 DeformableFeatureAggregationCuda
        self.output = self.model.detector.pts_bbox_head

        self.mul_list = []
        self.deform_list = []
        self.output_list = []

    def regist_forward_hooks(self):
        '''
        register forward hooks
        '''
        self.mulheadAttention.register_forward_hook(self.plot_corss_attention)
        # self.DeformableAttention.register_forward_hook(self.get_deformable_weight)
        self.output.register_forward_hook(self.get_ori_output)

    def plot_corss_attention(self, module, input, output):
        '''
        mulheadAttention module input function
        '''
        _, mulheadattention = output
        self.mul_list.append(mulheadattention)

    def get_deformable_weight(self, module, input, output):
        '''
        deformable module input function
        '''
        instance_feature, query_pos, feat_flatten, reference_points, spatial_flatten, level_start_index, pc_range, lidar2img_mat, img_metas = input
        bs, num_anchor = reference_points.shape[:2]
        reference_points = self.get_global_pos(reference_points, pc_range)
        key_points = reference_points.unsqueeze(-2) + module.learnable_fc(instance_feature).reshape(bs, num_anchor, -1, 3)

        weights = module._get_weights(instance_feature, query_pos, lidar2img_mat)

        features, point_2d = self.feature_sampling(feat_flatten, spatial_flatten, level_start_index, key_points, weights,
                                           lidar2img_mat, img_metas)

        output_cus = module.output_proj(features)
        output_cus = module.drop(output_cus) + instance_feature

        deform_dic = {'weights': weights.cpu(), 'point_2d': point_2d.cpu(), 'spatial_flatten': spatial_flatten.cpu()}

        self.deform_list.append(deform_dic)

    def feature_sampling(self, feat_flatten, spatial_flatten, level_start_index, key_points, weights, lidar2img_mat,
                         img_metas):
        bs, num_anchor, _ = key_points.shape[:3]

        pts_extand = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        points_2d = torch.matmul(lidar2img_mat[:, :, None, None], pts_extand[:, None, ..., None]).squeeze(-1)

        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        points_2d[..., 0:1] = points_2d[..., 0:1] / img_metas[0]['pad_shape'][0][1]
        points_2d[..., 1:2] = points_2d[..., 1:2] / img_metas[0]['pad_shape'][0][0]

        points_2d = points_2d.flatten(end_dim=1)  # [b*6, 900, 13, 2]
        points_2d = points_2d[:, :, None, None, :, :].repeat(1, 1, self.num_groups, self.num_levels, 1, 1)

        bn, num_value, _ = feat_flatten.size()
        feat_flatten = feat_flatten.reshape(bn, num_value, self.num_groups, -1)
        # attention_weights = weights * mask
        # output = MultiScaleDeformableAttnFunction.apply(
        #         feat_flatten, spatial_flatten, level_start_index, points_2d,
        #         weights, self.im2col_step)

        output = multi_scale_deformable_attn_pytorch(
            feat_flatten, spatial_flatten, points_2d,
            weights)

        output = output.reshape(bs, self.num_cams, num_anchor, -1)

        return output.sum(1), points_2d

    def get_ori_output(self, module, input, output):
        '''
        output of pts_head
        '''
        self.output_list.append(output)

    def get_global_pos(self, points, pc_range):
        points = points * (pc_range[3:6] - pc_range[0:3]) + pc_range[0:3]
        return points

    def handle_output(self, idx):
        '''
        the code from the StreamPETR
        just add the indexs of the query, the indexs denotes boxes order for the query
        '''
        preds_dicts = self.output_list[idx]
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]

        batch_size = all_cls_scores.size()[0]
        predictions_list = []

        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))

        num_samples = len(predictions_list)

        ret_list = []
        for i in range(num_samples):
            preds = predictions_list[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = LiDARInstance3DBoxes(bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            indexs = preds['indexs']
            ret_list.append([bboxes, scores, labels, indexs])

        bbox_results = [
            self.bbox3d2result(bboxes, scores, labels, indexs)
            for bboxes, scores, labels, indexs in ret_list
        ]

        bbox_list = [dict() for i in range(len(bbox_results))]
        for result_dict, pts_bbox in zip(bbox_list, bbox_results):
            result_dict['pts_bbox'] = pts_bbox

        if self.score_thr>0:
            bbox3ds_result = bbox_list[0]['pts_bbox']
            box3ds, scores, labels, indexs = bbox3ds_result['boxes_3d'], bbox3ds_result['scores_3d'], bbox3ds_result['labels_3d'], bbox3ds_result['indexs']
            inds = scores > self.score_thr
            box3ds = box3ds[inds, :]
            labels = labels[inds].cpu().numpy()
            classes = [self.CLASSES[i] for i in labels]
            scores = scores[inds]
            indexs = indexs[inds]

            bbox_list = [{"bboxes": box3ds, 'labels': labels, 'classes': classes, 'scores': scores, 'indexs': indexs}]

        return bbox_list

    def decode_single(self, cls_scores, bbox_preds):
        '''
        the code from the StreamPETR
        change the number query from 300 to 900 for get the all indexs
        '''
        max_num = self.max
        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = torch.div(indexs, self.num_classes, rounding_mode='floor')
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores
        final_preds = labels
        final_bbox = bbox_index
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)

            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            bbox_index = final_bbox[mask]
            assert boxes3d.shape[0] == self.max, f'the number of query shoule be {self.max}'
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
                'indexs': bbox_index
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def bbox3d2result(self, bboxes, scores, labels, indexs, attrs=None):
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            indexs=indexs.cpu())

        if attrs is not None:
            result_dict['attrs_3d'] = attrs.cpu()

        return result_dict

    def get_result_hooks(self):
        return self.mul_list, self.deform_list, self.output_list

    #vis about

    def vis_mul_head(self, mul_head_weight, new_result, save_path, idx):
        result_query = []
        classes = new_result['classes']
        for i in range(len(new_result['indexs'])):
            result_query.append(mul_head_weight[0][new_result['indexs'][i]][new_result['indexs']])

        result_query = torch.stack(result_query)

        fig, ax = plt.subplots()
        cax = ax.imshow(result_query, cmap='viridis', interpolation='nearest')
        cbar = plt.colorbar(cax, ax=ax)
        cbar.set_label('Weight Values')

        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))

        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        fig.savefig(f'{save_path}/{idx}.png')
        # plt.show()
