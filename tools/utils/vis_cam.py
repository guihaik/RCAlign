# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path
from functools import partial

import torch
import cv2
import mmcv
import numpy as np
from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmcv.parallel import MMDataParallel
from pytorch_grad_cam.utils.image import show_cam_on_image
from tools.utils.CustomHooks import CustomHooks
import matplotlib.pyplot as plt


from tools.utils.det_cam_visualizer import (DetAblationLayer, DetBoxScoreTarget, DetCAMModel,
                                            DetCAMVisualizer, FeatmapAM, reshape_transform, Det3DBoxScoreTarget)

try:
    from pytorch_grad_cam import AblationCAM, EigenCAM
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      '3rd party package pytorch_grad_cam.')

METHOD_MAP = {
    'ablationcam': AblationCAM,
    'eigencam': EigenCAM,
    'featmapam': FeatmapAM
}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--show', help='show 3dbox in image', default=True)
    parser.add_argument(
        '--method',
        default='ablationcam',
        help='Type of method to use, supports '
        f'{", ".join(list(METHOD_MAP.keys()))}.')
    parser.add_argument(
        '--target-layers',
        default=['img_neck'],
        nargs='+',
        type=str,
        help='The target layers to get CAM, if not set, the tool will '
        'specify the neck')
    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--topk',
        type=int,
        default=10,
        help='Topk of the predicted result to visualizer')
    parser.add_argument(
        '--max-reshape-shape',
        type=tuple,
        default=(32, 88),
        help='max reshape shapes. Its purpose is to save GPU memory. '
        'The activation map is scaled and then evaluated. '
        'If set to (-1, -1), it means no scaling.')
    parser.add_argument(
        '--norm-in-bbox',
        action='store_true',
        help='No norm in bbox of cam image')
    parser.add_argument(
        '--aug-smooth',
        default=False,
        action='store_true',
        help='Wether to use test time augmentation, default not to use')
    parser.add_argument(
        '--eigen-smooth',
        default=False,
        action='store_true',
        help='Reduce noise by taking the first principle componenet of '
        '``cam_weights*activations``')
    parser.add_argument('--out-dir', default=None, help='dir to output file')

    # Only used by AblationCAM
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='batch of inference of AblationCAM')
    parser.add_argument(
        '--ratio-channels-to-ablate',
        type=int,
        default=0.5,
        help='Making it much faster of AblationCAM. '
        'The parameter controls how many channels should be ablated')

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
    args = parser.parse_args()
    if args.method.lower() not in METHOD_MAP.keys():
        raise ValueError(f'invalid CAM type {args.method},'
                         f' supports {", ".join(list(METHOD_MAP.keys()))}.')

    return args

def init_model_cam(args, cfg):
    # build the model from a config file and a checkpoint file
    model = DetCAMModel(cfg, args.checkpoint, args.score_thr, device=args.device)

    if args.preview_model:
        print(model.detector)
        print('\n Please remove `--preview-model` to get the CAM.')
        return

    target_layers = []
    for target_layer in args.target_layers:
        try:
            target_layers.append(eval(f'model.detector.{target_layer}'))
        except Exception as e:
            print(model.detector)
            raise RuntimeError('layer does not exist', e)

    det_cam_visualizer = DetCAMVisualizer(
        args.method,
        model,
        target_layers,
        cfg,
        batch_size=args.batch_size,
        reshape_transform=partial(
            reshape_transform, max_shape=args.max_reshape_shape),
        ablation_layer=DetAblationLayer(),
        ratio_channels_to_ablate=args.ratio_channels_to_ablate)
    return model, det_cam_visualizer

def add_puugin(args, cfg):
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


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from plguin/xx, registry will be updated
    add_puugin(args, cfg)

    model, det_cam_visualizer = init_model_cam(args, cfg)

    cusHooks = CustomHooks(model)
    cusHooks.regist_forward_hooks()

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    if 'CLASSES' in model.checkpoint.get('meta', {}):
        model.CLASSES = model.checkpoint['meta']['CLASSES']
        model.COLORS = np.random.uniform(0, 255, size=(len(model.CLASSES), 3))
    else:
        model.CLASSES = dataset.CLASSES
        model.COLORS = np.random.uniform(0, 255, size=(len(model.CLASSES), 3))

    model.detector = MMDataParallel(model.detector, device_ids=[0])

    model.detector.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))


    everyquery_path = './pic/deform_everyquery'
    allquery_path = './pic/deform_allquery'
    savebox_path = './pic/deform_box'
    selattn_path = './pic/sefattn'
    cam_pth = './pic/cam'

    if not os.path.exists('./pic'):
        os.makedirs(everyquery_path)
        os.makedirs(allquery_path)
        os.makedirs(savebox_path)
        os.makedirs(selattn_path)
        os.makedirs(cam_pth)

    for i, data in enumerate(data_loader):
        model.input_data =data
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)[0]

            new_result = cusHooks.handle_output(i)[0]

            #vis mulheadattention
            mul_head_weight = cusHooks.mul_list[i].cpu()
            cusHooks.vis_mul_head(mul_head_weight, new_result, selattn_path, i)

            #vis deformabel attention
            deform_head = cusHooks.deform_list[i]
            deform_weights = deform_head['weights']
            deform_point2d = deform_head['point_2d']
            spatial_flatten = deform_head['spatial_flatten']

            det_deform_weights = deform_weights[:, new_result['indexs']]
            det_deform_point2d = deform_point2d[:, new_result['indexs']]

            num_cam, det_q, num_head, level, num_point, _ = det_deform_point2d.shape

            data_ = {}
            for key, value in data.items():
                data_[key] = value[0].data[0][0]

            for img_id in range(num_cam):
                img = data_['img'][img_id].squeeze().permute(1, 2, 0).numpy()
                img_nor_config = data_['img_metas']['img_norm_cfg']
                img = mmcv.imdenormalize(img, img_nor_config['mean'], img_nor_config['std'], img_nor_config['to_rgb'])
                lidar2img = data_['lidar2img'][img_id].numpy()
                point2d_all = []

                # img_3d, img_2d = model.draw_lidar_bbox_on_img(new_result['bboxes'], img, lidar2img, new_result['labels'])
                # cv2.imwrite(f'{savebox_path}/img3d_data{i}_view_{img_id}.png', img_3d)
                # cv2.imwrite(f'{savebox_path}/img2d_data{i}_view_{img_id}.png', img_2d)

                for q_id in range(det_q):
                    point2d_q_all = []
                    for head_id in range(num_head):
                        for level_id in range(level):
                            point2d = det_deform_point2d[img_id, q_id, head_id, level_id]
                            point2d = 2*point2d-1

                            #test point2d
                            # point2d = torch.tensor([[-0.5, 0.5], [-0.3, -0.4], [-5, 6]])

                            #point2d->grid, gird->feature map coors->img coors
                            #if the point2d is out of (point_range), the point2d should be (0,0)
                            point_range = torch.tensor([-1, -1, 1, 1])
                            mask = (point2d>= point_range[:2]).all(1)
                            mask &= (point2d<=point_range[2:]).all(1)
                            point2d[~mask] = 0
                            #获取空间尺度以及下采样的倍数用于之后向图像上变换
                            h, w = spatial_flatten[level_id]
                            sample_scale = img.shape[0] // h

                            #由于采样的时候中心点也会采样，所以最后产生的点是坐标的两倍，获取采样点
                            grid_x = torch.linspace(-1, 1, 2*w)
                            grid_y = torch.linspace(-1, 1, 2*h)
                            #获取到采样点中与预测得到的point2d中的点的相差最小的点的坐标，之后映射到特征坐标
                            grid_a, grid_b = torch.meshgrid(grid_x, point2d[:, 0])
                            diff_x = torch.abs(grid_a-grid_b)
                            grid_a, grid_b = torch.meshgrid(grid_y, point2d[:, 1])
                            diff_y = torch.abs(grid_a-grid_b)
                            #get the closest sample point index
                            near_x = torch.argmin(diff_x, dim=0)
                            near_y = torch.argmin(diff_y, dim=0)
                            #project the grid index to feature map
                            feat_x = torch.floor((grid_x[near_x] - point_range[0]) * w // 2)
                            feat_y = torch.floor((grid_y[near_y] - point_range[1]) * h // 2)
                            #project to the img coors
                            img_point2d = torch.stack([feat_x, feat_y], 0).t() * sample_scale
                            point2d_all.append(img_point2d)
                            point2d_q_all.append(img_point2d)
                    point2d_q_all = torch.cat(point2d_q_all, dim=0)


                    img_3d_q, img_2d_q = model.draw_lidar_bbox_on_img(new_result['bboxes'][q_id], img, lidar2img,[new_result['labels'][q_id]])
                    # cv2.imwrite(f'{savebox_path}/img3d_data{i}_view_{img_id}_q_{q_id}.png', img_3d)
                    # cv2.imwrite(f'{savebox_path}/img2d_data{i}_view_{img_id}_q_{q_id}.png', img_2d_q)

                    fig, ax = plt.subplots()
                    ax.scatter(point2d_q_all[:, 0], point2d_q_all[:, 1], c='red', marker='o', s=5)
                    ax.imshow(img_2d_q / 255)
                    plt.savefig(f'{everyquery_path}/data{i}_view{img_id}_{q_id}.png')

                point2d_all = torch.cat(point2d_all, dim=0)
                # img = cv2.imread(data_['img_metas']['filename'][0])
                img_3d, img_2d = model.draw_lidar_bbox_on_img(new_result['bboxes'], img, lidar2img, new_result['labels'])
                cv2.imwrite(f'{savebox_path}/img3d_data{i}_view_{img_id}.png', img_3d)
                cv2.imwrite(f'{savebox_path}/img2d_data{i}_view_{img_id}.png', img_2d)

                fig, ax = plt.subplots()
                ax.scatter(point2d_all[:, 0], point2d_all[:, 1], c='red', marker='o', s=5)
                ax.imshow(img / 255)
                plt.savefig(f'{allquery_path}/data{i}_view{img_id}.png')
                # plt.show()

            prog_bar.update()

    for i, data in enumerate(data_loader):
        model.input_data =data
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)[0]

            data_ = {}
            for key, value in data.items():
                data_[key] = value[0].data[0][0]

            targets = [Det3DBoxScoreTarget(dataset.data_infos[i])]
            grayscale_cam = det_cam_visualizer(
                data_['img'][0:1],
                targets=targets,
                aug_smooth=args.aug_smooth,
                eigen_smooth=args.eigen_smooth)
            for j in range(6):
                img = data_['img'][j].squeeze().permute(1, 2, 0).numpy()
                img_nor_config = data_['img_metas']['img_norm_cfg']
                img = mmcv.imdenormalize(img, img_nor_config['mean'], img_nor_config['std'], img_nor_config['to_rgb'])
                grayscale_cam_ = grayscale_cam[j]
                visualization = show_cam_on_image(img / 255, grayscale_cam_, use_rgb=False)
                cv2.imwrite(f'{cam_pth}/data{i}_view{j}.png', visualization)

            prog_bar.update()

if __name__ == '__main__':
    main()
