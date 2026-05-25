# # # from nuscenes.nuscenes import NuScenes
# # # from nuscenes.eval.detection.evaluate import NuScenesEval
# # # from nuscenes.eval.detection.data_classes import DetectionBox
# # # from nuscenes.eval.detection.config import config_factory
# # # from nuscenes.eval.common.data_classes import EvalBoxes
# # # import copy
# # # import os
# # #
# # #
# # # def filter_gt_by_attribute(nusc, nusc_eval, attribute_group='moving'):
# # #     """
# # #     根据属性过滤 GT boxes
# # #     attribute_group: 'moving' 或 'static'
# # #     """
# # #     original_eval_boxes = nusc_eval.gt_boxes
# # #     raw_gt_dict = original_eval_boxes.boxes
# # #     final_gt_boxes_dict = {}
# # #
# # #     print(f"Filtering GT for [{attribute_group}] objects...")
# # #
# # #     for sample_token, boxes in raw_gt_dict.items():
# # #         keep_boxes = []
# # #         for box in boxes:
# # #             # 计算速度
# # #             speed = (box.velocity[0] ** 2 + box.velocity[1] ** 2) ** 0.5
# # #             is_moving_speed = speed > 0.2  # 0.2 m/s 阈值
# # #
# # #             if attribute_group == 'moving':
# # #                 if is_moving_speed:
# # #                     keep_boxes.append(box)
# # #             elif attribute_group == 'static':
# # #                 if not is_moving_speed:
# # #                     keep_boxes.append(box)
# # #             else:
# # #                 keep_boxes.append(box)
# # #
# # #         final_gt_boxes_dict[sample_token] = keep_boxes
# # #
# # #     new_eval_boxes = EvalBoxes()
# # #     new_eval_boxes.boxes = final_gt_boxes_dict
# # #     nusc_eval.gt_boxes = new_eval_boxes
# # #     return nusc_eval
# # #
# # #
# # # def run_attribute_eval(nusc_version, dataroot, result_path, output_dir):
# # #     if not os.path.exists(output_dir):
# # #         os.makedirs(output_dir)
# # #
# # #     nusc = NuScenes(version=nusc_version, dataroot=dataroot, verbose=True)
# # #     cfg = config_factory("detection_cvpr_2019")
# # #
# # #     nusc_eval = NuScenesEval(nusc,
# # #                              config=cfg,
# # #                              result_path=result_path,
# # #                              eval_set='val',
# # #                              output_dir=output_dir,
# # #                              verbose=False)
# # #
# # #     original_gt = copy.deepcopy(nusc_eval.gt_boxes)
# # #
# # #     # ==========================
# # #     # Run 1: Moving AP
# # #     # ==========================
# # #     nusc_eval.gt_boxes = copy.deepcopy(original_gt)
# # #     filter_gt_by_attribute(nusc, nusc_eval, attribute_group='moving')
# # #
# # #     # metrics_moving 是一个 DetectionMetrics 对象
# # #     metrics_moving, _ = nusc_eval.evaluate()
# # #
# # #     print("\n>>> Moving Objects AP Results:")
# # #     # 【修复点 1】：使用 .label_aps 而不是 ['label_aps']
# # #     for cls in ['car', 'pedestrian', 'bicycle', 'bus', 'motorcycle', 'truck']:
# # #         if cls in metrics_moving.mean_dist_aps:
# # #             print(f"{cls}: {metrics_moving.mean_dist_aps[cls]:.4f}")
# # #     print('mAP:', metrics_moving.mean_ap)
# # #     # ==========================
# # #     # Run 2: Static AP
# # #     # ==========================
# # #     nusc_eval.gt_boxes = copy.deepcopy(original_gt)
# # #     filter_gt_by_attribute(nusc, nusc_eval, attribute_group='static')
# # #     metrics_static, _ = nusc_eval.evaluate()
# # #
# # #     print("\n>>> Static Objects AP Results:")
# # #     # 【修复点 2】：同上
# # #     for cls in ['car', 'pedestrian', 'bicycle', 'bus', 'motorcycle', 'truck']:
# # #         if cls in metrics_static.mean_dist_aps:
# # #             print(f"{cls}: {metrics_static.mean_dist_aps[cls]:.4f}")
# # #     print('mAP:', metrics_static.mean_ap)
# # #     # ==========================
# # #     # Run 3: AVE (Velocity Error)
# # #     # ==========================
# # #     nusc_eval.gt_boxes = copy.deepcopy(original_gt)
# # #     metrics_all, _ = nusc_eval.evaluate()
# # #
# # #     print("\n>>> Per-Class AVE (Velocity Error):")
# # #     # 【修复点 3】：使用 .label_ave 而不是 ['label_ave']
# # #     for cls, err in metrics_all.tp_errors.items():
# # #         print(f"{cls}: {err:.4f}")
# # #
# # #     print(metrics_all)
# # # if __name__ == "__main__":
# # #     path_root = 'work_dirs/pre_class/results_nusc/'
# # #     model_names = os.listdir(path_root)
# # #     for model_name in model_names:
# # #         print(f'process the {model_name}')
# # #         res_path = os.path.join(path_root, model_name, 'results_nusc.json')
# # #         out_path = os.path.join(path_root, model_name)
# # #         run_attribute_eval('v1.0-trainval', 'data/nuscenes', res_path, out_path)
# # #
# # #     # res_path = 'val/work_dirs/Ablation_Study/loss/ablation_study_r50_704_bs4_seq_loss_cosine_24e/Mon_Apr_29_01_28_06_2024/pts_bbox/results_nusc.json'
# # #     # out_path = 'work_dirs/pre_class'
# # #     #
# # #     # run_attribute_eval('v1.0-trainval', 'data/nuscenes', res_path, out_path)
# # #
# # #
# # from nuscenes.nuscenes import NuScenes
# # from nuscenes.eval.detection.evaluate import NuScenesEval
# # from nuscenes.eval.detection.data_classes import DetectionBox
# # from nuscenes.eval.detection.config import config_factory
# # from nuscenes.eval.common.data_classes import EvalBoxes
# # import copy
# # import os
# #
# #
# # def filter_gt_by_attribute(nusc, nusc_eval, attribute_group='moving'):
# #     """
# #     根据属性过滤 GT boxes
# #     attribute_group: 'moving' 或 'static'
# #     """
# #     original_eval_boxes = nusc_eval.gt_boxes
# #     raw_gt_dict = original_eval_boxes.boxes
# #     final_gt_boxes_dict = {}
# #
# #     print(f"Filtering GT for [{attribute_group}] objects...")
# #
# #     for sample_token, boxes in raw_gt_dict.items():
# #         keep_boxes = []
# #         for box in boxes:
# #             # 计算速度 (vx, vy)
# #             speed = (box.velocity[0] ** 2 + box.velocity[1] ** 2) ** 0.5
# #             is_moving_speed = speed > 0.2  # 0.2 m/s 阈值
# #
# #             if attribute_group == 'moving':
# #                 if is_moving_speed:
# #                     keep_boxes.append(box)
# #             elif attribute_group == 'static':
# #                 if not is_moving_speed:
# #                     keep_boxes.append(box)
# #             else:
# #                 keep_boxes.append(box)
# #
# #         final_gt_boxes_dict[sample_token] = keep_boxes
# #
# #     new_eval_boxes = EvalBoxes()
# #     new_eval_boxes.boxes = final_gt_boxes_dict
# #     nusc_eval.gt_boxes = new_eval_boxes
# #     return nusc_eval
# #
# #
# # def run_attribute_eval(nusc_version, dataroot, result_path, output_dir):
# #     if not os.path.exists(output_dir):
# #         os.makedirs(output_dir)
# #
# #     # 定义结果文件保存路径
# #     save_file_path = os.path.join(output_dir, 'metrics_summary.txt')
# #
# #     # 使用 'w' 模式打开文件，如果文件存在则覆盖
# #     with open(save_file_path, 'w') as f_out:
# #
# #         # 定义一个辅助函数，既打印到控制台，又写入文件
# #         def log_print(msg):
# #             print(msg)
# #             f_out.write(str(msg) + '\n')
# #
# #         log_print(f"Processing Result File: {result_path}")
# #
# #         nusc = NuScenes(version=nusc_version, dataroot=dataroot, verbose=True)
# #         cfg = config_factory("detection_cvpr_2019")
# #
# #         nusc_eval = NuScenesEval(nusc,
# #                                  config=cfg,
# #                                  result_path=result_path,
# #                                  eval_set='val',
# #                                  output_dir=output_dir,
# #                                  verbose=False)
# #
# #         original_gt = copy.deepcopy(nusc_eval.gt_boxes)
# #
# #         # ==========================
# #         # Run 1: Moving AP
# #         # ==========================
# #         nusc_eval.gt_boxes = copy.deepcopy(original_gt)
# #         filter_gt_by_attribute(nusc, nusc_eval, attribute_group='moving')
# #
# #         metrics_moving, _ = nusc_eval.evaluate()
# #
# #         log_print("\n>>> Moving Objects AP Results:")
# #         for cls in ['car', 'pedestrian', 'bicycle', 'bus', 'motorcycle', 'truck']:
# #             if cls in metrics_moving.mean_dist_aps:
# #                 log_print(f"{cls}: {metrics_moving.mean_dist_aps[cls]:.4f}")
# #         log_print(f'Moving mAP: {metrics_moving.mean_ap:.4f}')
# #
# #         # ==========================
# #         # Run 2: Static AP
# #         # ==========================
# #         nusc_eval.gt_boxes = copy.deepcopy(original_gt)
# #         filter_gt_by_attribute(nusc, nusc_eval, attribute_group='static')
# #         metrics_static, _ = nusc_eval.evaluate()
# #
# #         log_print("\n>>> Static Objects AP Results:")
# #         for cls in ['car', 'pedestrian', 'bicycle', 'bus', 'motorcycle', 'truck']:
# #             if cls in metrics_static.mean_dist_aps:
# #                 log_print(f"{cls}: {metrics_static.mean_dist_aps[cls]:.4f}")
# #         log_print(f'Static mAP: {metrics_static.mean_ap:.4f}')
# #
# #         # ==========================
# #         # Run 3: AVE (Velocity Error)
# #         # ==========================
# #         nusc_eval.gt_boxes = copy.deepcopy(original_gt)
# #         metrics_all, _ = nusc_eval.evaluate()
# #
# #         log_print("\n>>> Per-Class AVE (Velocity Error):")
# #
# #         # 【修正】：这里应该使用 label_ave 来获取各类别的速度误差
# #         # tp_errors 存储的是 {'ave': ..., 'ate': ...}，直接遍历是不对的
# #         # 【修复点 3】：使用 .label_ave 而不是 ['label_ave']
# #         if hasattr(metrics_all, 'label_ave'):
# #             for cls, err in metrics_all.tp_errors.items():
# #                 log_print(f"{cls}: {err:.4f}")
# #         else:
# #             # 兼容性处理，以防 label_ave 不存在 (虽然标准版都有)
# #             log_print("No label_ave found in metrics.")
# #
# #         # 如果你需要看详细的所有指标，可以打印整个 metrics 对象，但通常比较乱
# #         # log_print(str(metrics_all))
# #
# #         log_print("\n" + "=" * 50 + "\n")
# #
# #
# # if __name__ == "__main__":
# #     # 请修改这里的路径
# #     path_root = 'work_dirs/pre_class/results_nusc/'
# #     # path_root = '/你的/实际/路径/work_dirs/...'
# #
# #     if os.path.exists(path_root):
# #         model_names = os.listdir(path_root)
# #         for model_name in model_names:
# #             print(f'process the {model_name}')
# #
# #             # 假设每个模型文件夹下都有 results_nusc.json
# #             res_path = os.path.join(path_root, model_name, 'results_nusc.json')
# #             out_path = os.path.join(path_root, model_name, 'result_s_m')
# #
# #             if os.path.exists(res_path):
# #                 try:
# #                     run_attribute_eval('v1.0-trainval', 'data/nuscenes', res_path, out_path)
# #                 except Exception as e:
# #                     print(f"Error processing {model_name}: {e}")
# #             else:
# #                 print(f"Skipping {model_name}, results_nusc.json not found.")
# #     else:
# #         print(f"Path not found: {path_root}")
#
# from nuscenes.nuscenes import NuScenes
# from nuscenes.eval.detection.evaluate import NuScenesEval
# from nuscenes.eval.detection.data_classes import DetectionBox
# from nuscenes.eval.detection.config import config_factory
# from nuscenes.eval.common.data_classes import EvalBoxes
# import copy
# import os
#
#
# def filter_gt_by_attribute(nusc, nusc_eval, attribute_group='moving'):
#     """
#     根据属性过滤 GT boxes
#     attribute_group: 'moving' 或 'static'
#     """
#     original_eval_boxes = nusc_eval.gt_boxes
#     raw_gt_dict = original_eval_boxes.boxes
#     final_gt_boxes_dict = {}
#
#     print(f"Filtering GT for [{attribute_group}] objects...")
#
#     for sample_token, boxes in raw_gt_dict.items():
#         keep_boxes = []
#         for box in boxes:
#             # 计算速度 (vx, vy)
#             speed = (box.velocity[0] ** 2 + box.velocity[1] ** 2) ** 0.5
#             is_moving_speed = speed > 0.2  # 0.2 m/s 阈值
#
#             if attribute_group == 'moving':
#                 if is_moving_speed:
#                     keep_boxes.append(box)
#             elif attribute_group == 'static':
#                 if not is_moving_speed:
#                     keep_boxes.append(box)
#             else:
#                 keep_boxes.append(box)
#
#         final_gt_boxes_dict[sample_token] = keep_boxes
#
#     new_eval_boxes = EvalBoxes()
#     new_eval_boxes.boxes = final_gt_boxes_dict
#     nusc_eval.gt_boxes = new_eval_boxes
#     return nusc_eval
#
#
# def run_attribute_eval(nusc_version, dataroot, result_path, output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     # 定义结果文件保存路径
#     save_file_path = os.path.join(output_dir, 'metrics_summary1.txt')
#
#     # 定义所有 10 个类别
#     all_classes = [
#         'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
#         'pedestrian', 'motorcycle', 'bicycle',
#         'traffic_cone', 'barrier'
#     ]
#
#     # 使用 'w' 模式打开文件
#     with open(save_file_path, 'w') as f_out:
#
#         # 辅助打印函数
#         def log_print(msg):
#             print(msg)
#             f_out.write(str(msg) + '\n')
#
#         log_print(f"Processing Result File: {result_path}")
#
#         nusc = NuScenes(version=nusc_version, dataroot=dataroot, verbose=True)
#         cfg = config_factory("detection_cvpr_2019")
#
#         nusc_eval = NuScenesEval(nusc,
#                                  config=cfg,
#                                  result_path=result_path,
#                                  eval_set='val',
#                                  output_dir=output_dir,
#                                  verbose=False)
#
#         original_gt = copy.deepcopy(nusc_eval.gt_boxes)
#
#         # ==========================
#         # Run 1: Moving AP
#         # ==========================
#         nusc_eval.gt_boxes = copy.deepcopy(original_gt)
#         filter_gt_by_attribute(nusc, nusc_eval, attribute_group='moving')
#
#         metrics_moving, _ = nusc_eval.evaluate()
#
#         log_print("\n>>> Moving Objects AP Results:")
#         for cls in all_classes:
#             # 只有当该类别存在于结果中才打印 (被完全过滤掉的类别可能不在 keys 中)
#             if cls in metrics_moving.mean_dist_aps:
#                 log_print(f"{cls}: {metrics_moving.mean_dist_aps[cls]:.4f}")
#             else:
#                 # 如果因为全是静态物体被过滤没了，显示 N/A 或 0.0000
#                 log_print(f"{cls}: 0.0000 (No moving samples)")
#
#         log_print(f'Moving mAP: {metrics_moving.mean_ap:.4f}')
#
#         # ==========================
#         # Run 2: Static AP
#         # ==========================
#         nusc_eval.gt_boxes = copy.deepcopy(original_gt)
#         filter_gt_by_attribute(nusc, nusc_eval, attribute_group='static')
#         metrics_static, _ = nusc_eval.evaluate()
#
#         log_print("\n>>> Static Objects AP Results:")
#         for cls in all_classes:
#             if cls in metrics_static.mean_dist_aps:
#                 log_print(f"{cls}: {metrics_static.mean_dist_aps[cls]:.4f}")
#             else:
#                 log_print(f"{cls}: 0.0000 (No static samples)")
#
#         log_print(f'Static mAP: {metrics_static.mean_ap:.4f}')
#
#         # ==========================
#         # Run 3: AVE (Velocity Error)
#         # ==========================
#         nusc_eval.gt_boxes = copy.deepcopy(original_gt)
#         metrics_all, _ = nusc_eval.evaluate()
#
#         log_print("\n>>> Per-Class AVE (Velocity Error):")
#
#         if hasattr(metrics_all, 'tp_errors'):
#             for err_name, err in metrics_all.tp_errors:
#                 log_print(f"{err_name}: {metrics_all.tp_errors[err_name]:.4f}")
#             # for cls in all_classes:
#             #     # 注意：traffic_cone 和 barrier 通常不计算 AVE，因为它们是静态的
#             #     # 如果 key 存在则打印，不存在跳过或打印 N/A
#             #     if cls in metrics_all.label_ave:
#             #         log_print(f"{cls}: {metrics_all.label_ave[cls]:.4f}")
#             #     else:
#             #         # 对于 cone 和 barrier，通常没有 AVE 指标
#             #         pass
#         else:
#             log_print("No label_ave found in metrics.")
#
#         log_print("\n" + "=" * 50 + "\n")
#
#
# if __name__ == "__main__":
#     # 请修改这里的路径
#     path_root = 'work_dirs/pre_class/results_nusc/'
#
#     if os.path.exists(path_root):
#         model_names = os.listdir(path_root)
#         for model_name in model_names:
#             print(f'process the {model_name}')
#
#             res_path = os.path.join(path_root, model_name, 'results_nusc.json')
#             out_path = os.path.join(path_root, model_name)
#
#             if os.path.exists(res_path):
#                 try:
#                     run_attribute_eval('v1.0-trainval', 'data/nuscenes', res_path, out_path)
#                 except Exception as e:
#                     print(f"Error processing {model_name}: {e}")
#             else:
#                 print(f"Skipping {model_name}, results_nusc.json not found.")
#     else:
#         print(f"Path not found: {path_root}")


from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
import copy
import os


def filter_gt_by_attribute(nusc, nusc_eval, attribute_group='moving'):
    """
    根据属性过滤 GT boxes
    attribute_group: 'moving' 或 'static'
    """
    original_eval_boxes = nusc_eval.gt_boxes
    raw_gt_dict = original_eval_boxes.boxes
    final_gt_boxes_dict = {}

    print(f"Filtering GT for [{attribute_group}] objects...")

    for sample_token, boxes in raw_gt_dict.items():
        keep_boxes = []
        for box in boxes:
            # 计算速度 (vx, vy)
            speed = (box.velocity[0] ** 2 + box.velocity[1] ** 2) ** 0.5
            is_moving_speed = speed > 0.2  # 0.2 m/s 阈值

            if attribute_group == 'moving':
                if is_moving_speed:
                    keep_boxes.append(box)
            elif attribute_group == 'static':
                if not is_moving_speed:
                    keep_boxes.append(box)
            else:
                keep_boxes.append(box)

        final_gt_boxes_dict[sample_token] = keep_boxes

    new_eval_boxes = EvalBoxes()
    new_eval_boxes.boxes = final_gt_boxes_dict
    nusc_eval.gt_boxes = new_eval_boxes
    return nusc_eval


def filter_preds_by_attribute(nusc_eval, attribute_group='moving'):
    """
    【新增函数】根据预测速度过滤 Prediction boxes
    防止 static GT 被过滤后，static Prediction 变成 False Positive
    """
    original_eval_boxes = nusc_eval.pred_boxes  # 注意这里取 pred_boxes
    raw_pred_dict = original_eval_boxes.boxes
    final_pred_boxes_dict = {}

    print(f"Filtering Predictions for [{attribute_group}] objects...")

    for sample_token, boxes in raw_pred_dict.items():
        keep_boxes = []
        for box in boxes:
            # 计算预测速度 (vx, vy)
            # 预测框的 velocity 属性通常也是 (vx, vy)
            speed = (box.velocity[0] ** 2 + box.velocity[1] ** 2) ** 0.5
            is_moving_speed = speed > 0.2  # 阈值保持与 GT 一致

            if attribute_group == 'moving':
                # 只保留预测为动态的框
                if is_moving_speed:
                    keep_boxes.append(box)
            elif attribute_group == 'static':
                # 只保留预测为静态的框
                if not is_moving_speed:
                    keep_boxes.append(box)
            else:
                keep_boxes.append(box)

        final_pred_boxes_dict[sample_token] = keep_boxes

    new_eval_boxes = EvalBoxes()
    new_eval_boxes.boxes = final_pred_boxes_dict
    nusc_eval.pred_boxes = new_eval_boxes  # 赋值回 pred_boxes
    return nusc_eval


def run_attribute_eval(nusc_version, dataroot, result_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义结果文件保存路径
    save_file_path = os.path.join(output_dir, 'metrics_summary_filtered.txt')

    # 定义所有 10 个类别
    all_classes = [
        'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'pedestrian', 'motorcycle', 'bicycle',
        'traffic_cone', 'barrier'
    ]

    # 使用 'w' 模式打开文件
    with open(save_file_path, 'w') as f_out:

        # 辅助打印函数
        def log_print(msg):
            print(msg)
            f_out.write(str(msg) + '\n')

        log_print(f"Processing Result File: {result_path}")

        nusc = NuScenes(version=nusc_version, dataroot=dataroot, verbose=True)
        cfg = config_factory("detection_cvpr_2019")

        nusc_eval = NuScenesEval(nusc,
                                 config=cfg,
                                 result_path=result_path,
                                 eval_set='val',
                                 output_dir=output_dir,
                                 verbose=False)

        # 【关键步骤】同时备份原始 GT 和 原始 Pred
        original_gt = copy.deepcopy(nusc_eval.gt_boxes)
        original_preds = copy.deepcopy(nusc_eval.pred_boxes)

        # ==========================
        # Run 1: Moving AP
        # ==========================
        # 1. 重置 GT 和 Pred
        nusc_eval.gt_boxes = copy.deepcopy(original_gt)
        nusc_eval.pred_boxes = copy.deepcopy(original_preds)

        # 2. 双重过滤：过滤 GT + 过滤 Pred
        filter_gt_by_attribute(nusc, nusc_eval, attribute_group='moving')
        filter_preds_by_attribute(nusc_eval, attribute_group='moving')

        metrics_moving, _ = nusc_eval.evaluate()

        log_print("\n>>> Moving Objects AP Results:")
        for cls in all_classes:
            if cls in metrics_moving.mean_dist_aps:
                log_print(f"{cls}: {metrics_moving.mean_dist_aps[cls]:.4f}")
            else:
                log_print(f"{cls}: 0.0000 (No moving samples)")

        log_print(f'Moving mAP: {metrics_moving.mean_ap:.4f}')

        # ==========================
        # Run 2: Static AP
        # ==========================
        # 1. 重置 GT 和 Pred
        nusc_eval.gt_boxes = copy.deepcopy(original_gt)
        nusc_eval.pred_boxes = copy.deepcopy(original_preds)

        # 2. 双重过滤
        filter_gt_by_attribute(nusc, nusc_eval, attribute_group='static')
        filter_preds_by_attribute(nusc_eval, attribute_group='static')

        metrics_static, _ = nusc_eval.evaluate()

        log_print("\n>>> Static Objects AP Results:")
        for cls in all_classes:
            if cls in metrics_static.mean_dist_aps:
                log_print(f"{cls}: {metrics_static.mean_dist_aps[cls]:.4f}")
            else:
                log_print(f"{cls}: 0.0000 (No static samples)")

        log_print(f'Static mAP: {metrics_static.mean_ap:.4f}')

        # ==========================
        # Run 3: AVE (Velocity Error)
        # ==========================
        # 1. 恢复原始数据 (AVE 需要用所有数据计算)
        nusc_eval.gt_boxes = copy.deepcopy(original_gt)
        nusc_eval.pred_boxes = copy.deepcopy(original_preds)

        metrics_all, _ = nusc_eval.evaluate()

        log_print("\n>>> Per-Class AVE (Velocity Error):")

        # 修正 AVE 打印逻辑，使用 label_ave
        if hasattr(metrics_all, 'label_ave'):
            for cls in all_classes:
                if cls in metrics_all.label_ave:
                    log_print(f"{cls}: {metrics_all.label_ave[cls]:.4f}")
                else:
                    # 比如 barrier 通常没有 AVE
                    pass
        else:
            log_print("No label_ave found in metrics.")

        # 也可以打印一下总的 mAVE
        if hasattr(metrics_all, 'nd_score'):
            # 注意：metrics_all.tp_errors['ave'] 是所有类别的平均 AVE
            if 'ave' in metrics_all.tp_errors:
                log_print(f"Overall mAVE: {metrics_all.tp_errors['ave']:.4f}")

        log_print(f"NDS: {metrics_all.nd_score:.4f}")
        log_print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    # 请修改这里的路径
    path_root = 'work_dirs/pre_class/results_nusc/'

    if os.path.exists(path_root):
        model_names = os.listdir(path_root)
        for model_name in model_names:
            print(f'process the {model_name}')

            res_path = os.path.join(path_root, model_name, 'results_nusc.json')
            out_path = os.path.join(path_root, model_name)

            if os.path.exists(res_path):
                try:
                    run_attribute_eval('v1.0-trainval', 'data/nuscenes', res_path, out_path)
                except Exception as e:
                    print(f"Error processing {model_name}: {e}")
            else:
                print(f"Skipping {model_name}, results_nusc.json not found.")
    else:
        print(f"Path not found: {path_root}")