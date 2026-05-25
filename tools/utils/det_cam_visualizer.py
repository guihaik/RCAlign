# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import copy

import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torchvision
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)
from os import path as osp

import mmcv
from mmcv.image import tensor2imgs

try:
    from pytorch_grad_cam import AblationCAM, AblationLayer, EigenCAM
    from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      '3rd party package pytorch_grad_cam.')

from mmdet.apis import init_detector
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose


def reshape_transform(feats, max_shape=(20, 20)):
    """Reshape and aggregate feature maps when the input is a multi-layer
    feature map.

    Takes these tensors with different sizes, resizes them to a common shape,
    and concatenates them.
    """
    if isinstance(feats, torch.Tensor):
        feats = [feats]

    max_h = max([im.shape[-2] for im in feats])
    max_w = max([im.shape[-1] for im in feats])
    if -1 in max_shape:
        max_shape = (max_h, max_w)
    else:
        max_shape = (min(max_h, max_shape[0]), min(max_w, max_shape[1]))

    activations = []
    for feat in feats:
        activations.append(
            torch.nn.functional.interpolate(
                torch.abs(feat), max_shape, mode='bilinear'))

    activations = torch.cat(activations, axis=1)
    return activations


class DetCAMModel(nn.Module):
    """Wrap the mmdet model class to facilitate handling of non-tensor
    situations during inference."""

    def __init__(self, cfg, checkpoint, score_thr, device='cuda:0'):
        super().__init__()
        self.device = device
        self.score_thr = score_thr
        self.detector = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        self.checkpoint = load_checkpoint(self.detector, checkpoint, map_location='cpu')
        self.cfg = cfg
        self.show = False
        self.input_data = None
        self.img = None
        self.out_dir = 'cam'


    def draw_lidar_bbox3d_on_img(self, bboxes3d,
                                 raw_img,
                                 lidar2img_rt,
                                 color=(0, 255, 0),
                                 thickness=1):
        img = raw_img.copy()
        corners_3d = bboxes3d.corners
        num_bbox = corners_3d.shape[0]
        pts_4d = np.concatenate(
            [corners_3d.reshape(-1, 3),
             np.ones((num_bbox * 8, 1))], axis=-1)
        lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
        if isinstance(lidar2img_rt, torch.Tensor):
            lidar2img_rt = lidar2img_rt.cpu().numpy()
        pts_2d = pts_4d @ lidar2img_rt.T

        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

        return self.plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)

    def plot_rect3d_on_img(self, img,
                           num_rects,
                           rect_corners,
                           color=(0, 255, 0),
                           thickness=1):
        line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                        (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
        for i in range(num_rects):
            corners = rect_corners[i].astype(np.int32)
            for start, end in line_indices:
                cv2.line(img, (corners[start, 0], corners[start, 1]),
                         (corners[end, 0], corners[end, 1]), self.COLORS[color[i]], thickness,
                         cv2.LINE_AA)
            cv2.putText(img, self.CLASSES[color[i]], (corners[1, 0], corners[5, 1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS[color[i]], 2,
                        lineType=cv2.LINE_AA)
        return img.astype(np.uint8)


    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        assert self.input_data is not None
        result = self.detector(return_loss=False, rescale=True, **self.input_data)

        bbox3ds_result = result[0]['pts_bbox']
        box3ds, scores, labels = bbox3ds_result['boxes_3d'], bbox3ds_result['scores_3d'], bbox3ds_result['labels_3d']

        if self.score_thr > 0:
            inds = scores > self.score_thr
            box3ds = box3ds[inds, :]
            labels = labels[inds].cpu().numpy()
            classes = [self.CLASSES[i] for i in labels]
            scores = scores[inds]
        return [{"bboxes": box3ds, 'labels': labels, 'classes': classes, 'scores':scores}]


class DetAblationLayer(AblationLayer):

    def __init__(self):
        super(DetAblationLayer, self).__init__()
        self.activations = None

    def set_next_batch(self, input_batch_index, activations,
                       num_channels_to_ablate):
        """Extract the next batch member from activations, and repeat it
        num_channels_to_ablate times."""
        if isinstance(activations, torch.Tensor):
            return super(DetAblationLayer,
                         self).set_next_batch(input_batch_index, activations,
                                              num_channels_to_ablate)

        self.activations = []
        for activation in activations:
            activation = activation[
                input_batch_index, :, :, :].clone().unsqueeze(0)
            self.activations.append(
                activation.repeat(num_channels_to_ablate, 1, 1, 1))

    def __call__(self, x):
        """Go over the activation indices to be ablated, stored in
        self.indices.

        Map between every activation index to the tensor in the Ordered Dict
        from the FPN layer.
        """
        result = self.activations

        if isinstance(result, torch.Tensor):
            return super(DetAblationLayer, self).__call__(x)

        channel_cumsum = np.cumsum([r.shape[1] for r in result])
        num_channels_to_ablate = result[0].size(0)  # batch
        for i in range(num_channels_to_ablate):
            pyramid_layer = bisect.bisect_right(channel_cumsum,
                                                self.indices[i])
            if pyramid_layer > 0:
                index_in_pyramid_layer = self.indices[i] - channel_cumsum[
                    pyramid_layer - 1]
            else:
                index_in_pyramid_layer = self.indices[i]
            result[pyramid_layer][i, index_in_pyramid_layer, :, :] = -1000
        return result


class DetCAMVisualizer:
    """mmdet cam visualization class.

    Args:
        method (str):  CAM method. Currently supports
           `ablationcam`,`eigencam` and `featmapam`.
        model (nn.Module): MMDet model.
        target_layers (list[torch.nn.Module]): The target layers
            you want to visualize.
        ablation_layer (torch.nn.Module): The ablation layer. Only
            used by AblationCAM method. Defaults to None.
        reshape_transform (Callable, optional): Function of Reshape
            and aggregate feature maps. Defaults to None.
        batch_size (int): Batch of inference of AblationCAM. Only
            used by AblationCAM method. Defaults to 1.
        ratio_channels_to_ablate (float): The parameter controls how
            many channels should be ablated. Only used by
            AblationCAM method. Defaults to 0.1.
    """

    def __init__(self,
                 method,
                 model,
                 target_layers,
                 cfg,
                 ablation_layer=None,
                 reshape_transform=None,
                 batch_size=1,
                 ratio_channels_to_ablate=0.1):
        if method == 'ablationcam':
            self.cam = AblationCAM(
                model,
                target_layers,
                use_cuda=False,
                reshape_transform=reshape_transform,
                batch_size=batch_size,
                ablation_layer=ablation_layer,
                ratio_channels_to_ablate=ratio_channels_to_ablate)
        elif method == 'eigencam':
            self.cam = EigenCAM(
                model,
                target_layers,
                use_cuda=False,
                reshape_transform=reshape_transform,
            )
        elif method == 'featmapam':
            self.cam = FeatmapAM(
                model,
                target_layers,
                use_cuda=False,
                reshape_transform=reshape_transform,
            )
        else:
            raise NotImplementedError(
                f'{method} cam calculation method is not supported')

        self.classes = cfg.class_names
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def __call__(self, img, targets, aug_smooth=False, eigen_smooth=False):
        # img = img.cuda()
        return self.cam(img, targets, aug_smooth, eigen_smooth)

    def show_cam(self,
                 image,
                 boxes,
                 labels,
                 grayscale_cam,
                 with_norm_in_bboxes=False):
        """Normalize the CAM to be in the range [0, 1] inside every bounding
        boxes, and zero outside of the bounding boxes."""
        if with_norm_in_bboxes is True:
            boxes = boxes.astype(np.int32)
            renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
            images = []
            for x1, y1, x2, y2 in boxes:
                img = renormalized_cam * 0
                img[y1:y2,
                    x1:x2] = scale_cam_image(grayscale_cam[y1:y2,
                                                           x1:x2].copy())
                images.append(img)

            renormalized_cam = np.max(np.float32(images), axis=0)
            renormalized_cam = scale_cam_image(renormalized_cam)
        else:
            renormalized_cam = grayscale_cam

        cam_image_renormalized = show_cam_on_image(
            image / 255, renormalized_cam, use_rgb=False)

        image_with_bounding_boxes = self._draw_boxes(boxes, labels,
                                                     cam_image_renormalized)
        return image_with_bounding_boxes

    def _draw_boxes(self, boxes, labels, image):
        for i, box in enumerate(boxes):
            label = labels[i]
            color = self.COLORS[label]
            cv2.rectangle(image, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), color, 2)
            cv2.putText(
                image,
                self.classes[label], (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                lineType=cv2.LINE_AA)
        return image

class Det3DBoxScoreTarget:
    def __init__(self, data_info, iou_threshold=0.5, device='cuda:0'):
        self.data_inf = data_info
        self.iou_threshold = iou_threshold
        self.device = device

    def __call__(self, results):
        output = torch.tensor([0], device=self.device)

        if len(results["bboxes"]) == 0:
            return output

        for i in self.scores:
            output+=self.scores[i]
        return output

class DetBoxScoreTarget:
    """For every original detected bounding box specified in "bboxes",
    assign a score on how the current bounding boxes match it,
        1. In Bbox IoU
        2. In the classification score.
        3. In Mask IoU if ``segms`` exist.

    If there is not a large enough overlap, or the category changed,
    assign a score of 0.

    The total score is the sum of all the box scores.
    """

    def __init__(self,
                 bboxes,
                 labels,
                 segms=None,
                 match_iou_thr=0.5,
                 device='cuda:0'):
        assert len(bboxes) == len(labels)
        self.focal_bboxes = torch.from_numpy(bboxes).to(device=device)
        self.focal_labels = labels
        if segms is not None:
            assert len(bboxes) == len(segms)
            self.focal_segms = torch.from_numpy(segms).to(device=device)
        else:
            self.focal_segms = [None] * len(labels)
        self.match_iou_thr = match_iou_thr

        self.device = device

    def __call__(self, results):
        output = torch.tensor([0], device=self.device)
        if len(results["bboxes"]) == 0:
            return output

        pred_bboxes = torch.from_numpy(results["bboxes"]).to(self.device)
        pred_labels = results["labels"]
        pred_segms = results["segms"]

        if pred_segms is not None:
            pred_segms = torch.from_numpy(pred_segms).to(self.device)

        for focal_box, focal_label, focal_segm in zip(self.focal_bboxes,
                                                      self.focal_labels,
                                                      self.focal_segms):
            ious = torchvision.ops.box_iou(focal_box[None],
                                           pred_bboxes[..., :4])
            index = ious.argmax()
            if ious[0, index] > self.match_iou_thr and pred_labels[
                    index] == focal_label:
                # TODO: Adaptive adjustment of weights based on algorithms
                score = ious[0, index] + pred_bboxes[..., 4][index]
                output = output + score

                if focal_segm is not None and pred_segms is not None:
                    segms_score = (focal_segm * pred_segms[index]).sum() / (
                        focal_segm.sum() + pred_segms[index].sum() + 1e-7)
                    output = output + segms_score
        return output


class FeatmapAM(EigenCAM):
    """Visualize Feature Maps.

    Visualize the (B,C,H,W) feature map averaged over the channel dimension.
    """

    def __init__(self,
                 model,
                 target_layers,
                 use_cuda=False,
                 reshape_transform=None):
        super(FeatmapAM, self).__init__(model, target_layers, use_cuda,
                                        reshape_transform)

    def get_cam_image(self, input_tensor, target_layer, target_category,
                      activations, grads, eigen_smooth):
        return np.mean(activations, axis=1)
