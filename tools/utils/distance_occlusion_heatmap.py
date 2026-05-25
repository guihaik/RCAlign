# import os
# import json
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.data_classes import Box
# from pyquaternion import Quaternion
#
#
# def generate_heatmap_data(nusc, result_path, dist_bins, match_thresh=2.0, score_thresh=0.1):
#     """
#     修正版: 增加了类别匹配和置信度过滤
#     score_thresh: 过滤掉置信度低于 0.1 的预测框 (防止 noise matching)
#     """
#     with open(result_path, 'r') as f:
#         predictions = json.load(f)['results']
#
#     vis_levels = [1, 2, 3, 4]
#     rows = len(vis_levels)
#     cols = len(dist_bins) - 1
#     matrix_gt = np.zeros((rows, cols))
#     matrix_tp = np.zeros((rows, cols))
#
#     print(f"Processing {result_path} ...")
#
#     # 简单的类别映射 (nuScenes GT name -> detection name)
#     # 根据你的模型输出类别进行调整，这里列举了常见的
#     name_map = {
#         'movable_object.barrier': 'barrier',
#         'vehicle.bicycle': 'bicycle',
#         'vehicle.bus.bendy': 'bus',
#         'vehicle.bus.rigid': 'bus',
#         'vehicle.car': 'car',
#         'vehicle.construction': 'construction_vehicle',
#         'vehicle.motorcycle': 'motorcycle',
#         'human.pedestrian.adult': 'pedestrian',
#         'human.pedestrian.child': 'pedestrian',
#         'human.pedestrian.construction_worker': 'pedestrian',
#         'human.pedestrian.police_officer': 'pedestrian',
#         'movable_object.trafficcone': 'traffic_cone',
#         'vehicle.trailer': 'trailer',
#         'vehicle.truck': 'truck'
#     }
#
#     for sample_token in predictions.keys():
#         try:
#             sample = nusc.get('sample', sample_token)
#         except:
#             continue
#
#         # 1. 获取 Ego Pose
#         sd_token = sample['data']['LIDAR_TOP']
#         sd_record = nusc.get('sample_data', sd_token)
#         ego_pose = nusc.get('ego_pose', sd_record['ego_pose_token'])
#         ego_trans = np.array(ego_pose['translation'])
#
#         # 2. 预处理当前帧的预测框 (过滤 Score + 解析 Class)
#         raw_preds = predictions[sample_token]
#         valid_preds = []
#         for p in raw_preds:
#             # 【修正点 1】过滤低置信度
#             if p['detection_score'] < score_thresh:
#                 continue
#             valid_preds.append({
#                 'loc': np.array(p['translation'][:2]),
#                 'name': p['detection_name']
#             })
#
#         # 如果当前帧没有有效预测，后面所有 GT 都是 FN
#         no_preds = (len(valid_preds) == 0)
#
#         for ann_token in sample['anns']:
#             ann = nusc.get('sample_annotation', ann_token)
#
#             # 【修正点 2】只统计我们要检测的类别 (例如只看车、人等)
#             # 如果你的模型只检测 10 类，其他 GT 应该跳过
#             gt_cat = ann['category_name']
#             det_name = None
#
#             # 查找映射，如果 GT 类别不在映射表中（比如 animal），跳过不统计
#             for k, v in name_map.items():
#                 if gt_cat.startswith(k):  # startswith 处理子类
#                     det_name = v
#                     break
#
#             if det_name is None:
#                 continue  # 这个 GT 不是我们要检测的目标类别
#
#             # 计算 GT 距离和遮挡
#             gt_loc = np.array(ann['translation'])
#             dist_vec = gt_loc[:2] - ego_trans[:2]
#             gt_dist = np.linalg.norm(dist_vec)
#             vis_token = int(nusc.get('visibility', ann['visibility_token'])['token'])
#
#             # 确定 Bin 位置
#             col_idx = -1
#             for i in range(len(dist_bins) - 1):
#                 if dist_bins[i] <= gt_dist < dist_bins[i + 1]:
#                     col_idx = i
#                     break
#             if col_idx == -1: continue
#             if vis_token not in vis_levels: continue
#             row_idx = vis_levels.index(vis_token)
#
#             # 3. 匹配逻辑
#             matrix_gt[row_idx, col_idx] += 1
#
#             if not no_preds:
#                 is_detected = False
#                 # 遍历所有有效预测
#                 for pred in valid_preds:
#                     # 【修正点 3】必须 类别匹配 且 距离足够近
#                     if pred['name'] == det_name:
#                         dist = np.linalg.norm(pred['loc'] - gt_loc[:2])
#                         if dist < match_thresh:
#                             is_detected = True
#                             break  # 找到一个就行 (Greedy Match)
#
#                 if is_detected:
#                     matrix_tp[row_idx, col_idx] += 1
#
#     with np.errstate(divide='ignore', invalid='ignore'):
#         recall_matrix = matrix_tp / matrix_gt
#         recall_matrix = np.nan_to_num(recall_matrix)
#
#     return recall_matrix * 100
#
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
#
# def plot_heatmap(data, dist_bins, title, output_path):
#     # 1. 调大画布，防止文字被切掉
#     plt.figure(figsize=(14, 10))
#
#     x_labels = [f"{dist_bins[i]}-{dist_bins[i + 1]}m" for i in range(len(dist_bins) - 1)]
#     y_labels = [
#         "60-100%",
#         "40-60%",
#         "20-40%",
#         "0-20% "
#     ]
#
#     # 2. 绘制热力图
#     # annot_kws 控制格子内的数字字体
#     ax = sns.heatmap(data, annot=True, fmt=".1f", cmap="YlOrRd",
#                      xticklabels=x_labels, yticklabels=y_labels,
#                      vmin=0, vmax=80,
#                      annot_kws={"size": 20, "weight": "bold"})
#
#     # 3. 强制设置 Y 轴标签水平显示 (关键修改)
#     # rotation=0: 水平
#     # va='center': 垂直居中对齐
#     ax.set_yticklabels(y_labels, rotation=90, fontsize=16, fontweight='bold', va='center')
#
#     # 4. 强制设置 X 轴标签水平显示 (防止倾斜)
#     ax.set_xticklabels(x_labels, rotation=0, fontsize=16, fontweight='bold')
#
#     # 5. 设置坐标轴标题
#     plt.xlabel("Distance to Ego", fontsize=20, fontweight='bold', labelpad=20)
#     plt.ylabel("Occlusion  Level", fontsize=20, fontweight='bold', labelpad=20)
#
#     # 6. 调整 Colorbar 字体
#     cbar = ax.collections[0].colorbar
#     cbar.ax.tick_params(labelsize=16)
#
#     # 7. 关键：自动调整布局，防止左侧长文字被切掉
#     plt.tight_layout()
#
#     plt.savefig(output_path, dpi=300)
#     plt.close()
#     print(f"Heatmap saved to {output_path}")
#
#
# # ================= 主程序保持不变 =================
# if __name__ == "__main__":
#     nusc_version = 'v1.0-trainval'
#     dataroot = 'data/nuscenes'
#
#     # 你的路径
#     res_path_ours = 'work_dirs/pre_class/results_nusc/RCAlign_24e/results_nusc.json'
#     res_path_baseline = 'work_dirs/pre_class/results_nusc/StreamPETR/results_nusc.json'
#
#     dist_bins = [0, 10, 20, 30, 40, 50]
#
#     nusc = NuScenes(version=nusc_version, dataroot=dataroot, verbose=True)
#
#     # 1. 计算 Ours
#     heatmap_ours = generate_heatmap_data(nusc, res_path_ours, dist_bins)
#     plot_heatmap(heatmap_ours, dist_bins, "Ours: Recall (%)", "heatmap_ours.png")
#
#     # 2. 计算 Baseline 并画 Delta
#     if os.path.exists(res_path_baseline):
#         heatmap_base = generate_heatmap_data(nusc, res_path_baseline, dist_bins)
#         plot_heatmap(heatmap_base, dist_bins, "Baseline: Recall (%)", "heatmap_baseline.png")
#
#         heatmap_delta = heatmap_ours - heatmap_base
#
#         plt.figure(figsize=(8, 6))
#         x_labels = [f"{dist_bins[i]}-{dist_bins[i + 1]}m" for i in range(len(dist_bins) - 1)]
#         y_labels = ["0-40% (Heavy)", "40-60%", "60-80%", "80-100% (None)"]
#
#         # Delta 建议 vmin/vmax 设为自动或者手动指定范围 (如 -10 到 10) 以便突出对比
#         sns.heatmap(heatmap_delta, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
#                     xticklabels=x_labels, yticklabels=y_labels)
#         plt.title("Improvement: Ours - Baseline (%)")
#         plt.xlabel("Distance")
#         plt.ylabel("Visibility")
#         plt.tight_layout()
#         plt.savefig("heatmap_delta.png", dpi=300)
#         print("Delta heatmap saved.")

# import os
# import json
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from nuscenes.nuscenes import NuScenes
#
#
# # ================= AP 计算核心逻辑 =================
#
# def calculate_ap(gt_boxes, pred_boxes, match_dist_thresh=2.0):
#     """
#     计算单个类别的 Average Precision (AP)
#     gt_boxes: list of {'loc': np.array, 'token': str}
#     pred_boxes: list of {'loc': np.array, 'token': str, 'score': float}
#     """
#     if len(gt_boxes) == 0:
#         return 0.0
#     if len(pred_boxes) == 0:
#         return 0.0
#
#     # 1. 按分数降序排列预测框
#     pred_boxes = sorted(pred_boxes, key=lambda x: x['score'], reverse=True)
#
#     nd = len(pred_boxes)
#     tp = np.zeros(nd)
#     fp = np.zeros(nd)
#
#     # 记录 GT 是否被匹配过，防止重复匹配
#     # key: sample_token_gt_index (为了简化，我们在外部处理好唯一标识，这里用 set 记录已匹配的 GT index)
#     gt_matched = set()
#
#     # 2. 遍历预测框进行匹配
#     for i, pred in enumerate(pred_boxes):
#         # 找出同一帧内的所有 GT
#         # 优化：实际场景中应该先 filter token，这里列表较短直接遍历
#         candidates = [idx for idx, gt in enumerate(gt_boxes) if gt['token'] == pred['token']]
#
#         best_dist = float('inf')
#         best_gt_idx = -1
#
#         # 寻找最近的 GT
#         for gt_idx in candidates:
#             # 如果这个 GT 已经被更高分的预测框匹配了，跳过
#             if gt_idx in gt_matched:
#                 continue
#
#             dist = np.linalg.norm(pred['loc'] - gt_boxes[gt_idx]['loc'])
#             if dist < best_dist:
#                 best_dist = dist
#                 best_gt_idx = gt_idx
#
#         # 判定 TP/FP
#         if best_dist < match_dist_thresh:
#             tp[i] = 1.0
#             gt_matched.add(best_gt_idx)
#         else:
#             fp[i] = 1.0
#
#     # 3. 计算 Cumulative Precision / Recall
#     acc_tp = np.cumsum(tp)
#     acc_fp = np.cumsum(fp)
#
#     eps = 1e-6
#     rec = acc_tp / len(gt_boxes)
#     prec = acc_tp / (acc_tp + acc_fp + eps)
#
#     # 4. 计算 AP (使用 VOC 2010+ 也就是 COCO 风格的平滑计算，或者是简单的 AUC)
#     # 这里使用 11-point interpolation 或者 All-points interpolation
#     # 为了简单且鲁棒，使用 All-points interpolation (Area Under Curve)
#     ap = 0.0
#     # 在 precision 序列末尾补 0，recall 前补 0，方便计算
#     mrec = np.concatenate(([0.], rec, [1.]))
#     mpre = np.concatenate(([0.], prec, [0.]))
#
#     # Compute the precision envelope
#     for i in range(mpre.size - 1, 0, -1):
#         mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
#
#     # Integrate area under curve
#     i = np.where(mrec[1:] != mrec[:-1])[0]
#     ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
#
#     return ap
#
#
# # ================= 数据处理与热力图 =================
#
# def generate_map_heatmap_data(nusc, result_path, dist_bins, match_thresh=2.0, score_thresh=0.1):
#     with open(result_path, 'r') as f:
#         predictions = json.load(f)['results']
#
#     vis_levels = [1, 2, 3, 4]
#     rows = len(vis_levels)
#     cols = len(dist_bins) - 1
#
#     # 初始化数据桶
#     # gts_bucket[row][col][class_name] = list of boxes
#     # preds_bucket[col][class_name] = list of boxes (注意：预测框没有遮挡属性，只按距离分桶)
#     gts_bucket = [[{} for _ in range(cols)] for _ in range(rows)]
#     preds_bucket = [{} for _ in range(cols)]
#
#     # 定义关注的类别
#     target_classes = [
#         'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
#         'pedestrian', 'motorcycle', 'bicycle', 'barrier', 'traffic_cone'
#     ]
#     # 类别映射表
#     name_map = {
#         'vehicle.car': 'car', 'vehicle.truck': 'truck', 'vehicle.bus.rigid': 'bus',
#         'vehicle.bus.bendy': 'bus', 'vehicle.trailer': 'trailer',
#         'vehicle.construction': 'construction_vehicle', 'human.pedestrian.adult': 'pedestrian',
#         'human.pedestrian.child': 'pedestrian', 'human.pedestrian.construction_worker': 'pedestrian',
#         'human.pedestrian.police_officer': 'pedestrian', 'vehicle.motorcycle': 'motorcycle',
#         'vehicle.bicycle': 'bicycle', 'static_object.bicycle_rack': 'bicycle',
#         'static_object.barrier': 'barrier', 'static_object.traffic_cone': 'traffic_cone'
#     }
#
#     print(f"Collecting data from {result_path} ...")
#
#     for sample_token in predictions.keys():
#         try:
#             sample = nusc.get('sample', sample_token)
#         except:
#             continue
#
#         # 1. 获取 Ego Pose
#         sd_token = sample['data']['LIDAR_TOP']
#         sd_record = nusc.get('sample_data', sd_token)
#         ego_pose = nusc.get('ego_pose', sd_record['ego_pose_token'])
#         ego_trans = np.array(ego_pose['translation'])
#
#         # 2. 处理预测框 (按距离分桶)
#         raw_preds = predictions[sample_token]
#         for p in raw_preds:
#             if p['detection_score'] < score_thresh: continue
#
#             # 这里的类别如果不在我们关注的列表里，也可以跳过，或者为了严谨保留作为FP
#             # 简单起见，只保留关注类别
#             p_class = p['detection_name']
#             if p_class not in target_classes: continue
#
#             # 计算 Pred 距离
#             loc = np.array(p['translation'])  # Global
#             dist = np.linalg.norm(loc[:2] - ego_trans[:2])
#
#             # 放入对应的距离桶
#             col_idx = -1
#             for i in range(len(dist_bins) - 1):
#                 if dist_bins[i] <= dist < dist_bins[i + 1]:
#                     col_idx = i
#                     break
#
#             if col_idx != -1:
#                 if p_class not in preds_bucket[col_idx]:
#                     preds_bucket[col_idx][p_class] = []
#                 preds_bucket[col_idx][p_class].append({
#                     'loc': loc[:2],
#                     'token': sample_token,
#                     'score': p['detection_score']
#                 })
#
#         # 3. 处理 GT (按距离 + 遮挡 分桶)
#         for ann_token in sample['anns']:
#             ann = nusc.get('sample_annotation', ann_token)
#
#             # 类别映射
#             gt_cat_raw = ann['category_name']
#             gt_class = None
#             for k, v in name_map.items():
#                 if gt_cat_raw.startswith(k):
#                     gt_class = v
#                     break
#             if gt_class not in target_classes: continue
#
#             # GT 距离 & 遮挡
#             gt_loc = np.array(ann['translation'])
#             gt_dist = np.linalg.norm(gt_loc[:2] - ego_trans[:2])
#             vis_token = int(nusc.get('visibility', ann['visibility_token'])['token'])
#
#             # 确定 Grid 位置
#             col_idx = -1
#             for i in range(len(dist_bins) - 1):
#                 if dist_bins[i] <= gt_dist < dist_bins[i + 1]:
#                     col_idx = i
#                     break
#
#             if col_idx != -1 and vis_token in vis_levels:
#                 row_idx = vis_levels.index(vis_token)
#
#                 if gt_class not in gts_bucket[row_idx][col_idx]:
#                     gts_bucket[row_idx][col_idx][gt_class] = []
#
#                 gts_bucket[row_idx][col_idx][gt_class].append({
#                     'loc': gt_loc[:2],
#                     'token': sample_token
#                 })
#
#     # 4. 计算每个 Grid 的 mAP
#     print("Calculating mAP for each bin...")
#     map_matrix = np.zeros((rows, cols))
#
#     for r in range(rows):
#         for c in range(cols):
#             aps = []
#             # 遍历每一个类别，分别计算 AP
#             for cls in target_classes:
#                 gts = gts_bucket[r][c].get(cls, [])
#
#                 # 【重点】Preds 取自对应的距离桶 (preds_bucket[c])
#                 # 因为 Prediction 不区分遮挡，所以同一距离下的 Pred 对该距离下所有遮挡等级的 GT 有效
#                 preds = preds_bucket[c].get(cls, [])
#
#                 if len(gts) > 0:
#                     ap = calculate_ap(gts, preds, match_dist_thresh=match_thresh)
#                     aps.append(ap)
#                 # 如果这个 bin 里没有该类别的 GT，通常不计入 mAP 的平均（或者计为0，视具体定义）
#                 # 这里我们只平均“存在 GT 的类别”的 AP
#
#             if len(aps) > 0:
#                 map_matrix[r, c] = np.mean(aps) * 100  # 转百分比
#             else:
#                 map_matrix[r, c] = 0.0  # 无 GT 区域
#
#     return map_matrix
#
#
# def plot_heatmap(data, dist_bins, title, output_path):
#     plt.figure(figsize=(8, 6))
#
#     x_labels = [f"{dist_bins[i]}-{dist_bins[i + 1]}m" for i in range(len(dist_bins) - 1)]
#     # vis_level: 1=Heavy, 4=None
#     y_labels = ["0-40% (Heavy)", "40-60%", "60-80%", "80-100% (None)"]
#
#     ax = sns.heatmap(data, annot=True, fmt=".1f", cmap="YlGnBu",
#                      xticklabels=x_labels, yticklabels=y_labels,
#                      vmin=0, vmax=data.max())  # mAP 最大值自适应或者设为 60/80
#
#     plt.title(title)
#     plt.xlabel("Distance to Ego")
#     plt.ylabel("Visibility (Occlusion)")
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()
#     print(f"Heatmap saved to {output_path}")
#
#
# # ================= 主程序 =================
# if __name__ == "__main__":
#     nusc_version = 'v1.0-trainval'
#     dataroot = 'data/nuscenes'
#
#     # 你的路径
#     res_path_ours = 'work_dirs/pre_class/results_nusc/RCAlign_24e/results_nusc.json'
#     # res_path_baseline = '...'
#
#     dist_bins = [0, 20, 40, 60, 80]
#
#     nusc = NuScenes(version=nusc_version, dataroot=dataroot, verbose=True)
#
#     # 计算 Ours mAP
#     heatmap_ours = generate_map_heatmap_data(nusc, res_path_ours, dist_bins, match_thresh=2.0)
#     plot_heatmap(heatmap_ours, dist_bins, "Ours: mAP (%)", "heatmap_map_ours_map.png")

import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes


# ================= 1. AP 计算核心逻辑 (修正版) =================

def calculate_masked_ap(gt_boxes_all, pred_boxes, target_tokens, match_dist_thresh):
    """
    计算 AP，引入 Ignore 机制。
    :param gt_boxes_all: 当前距离段内所有的 GT (无论遮挡等级)
    :param pred_boxes: 当前距离段内所有的 Pred
    :param target_tokens: 当前遮挡等级(bin)需要评估的 GT token 集合 (Set)
    :param match_dist_thresh: 距离阈值
    """
    if len(target_tokens) == 0:
        return 0.0
    if len(pred_boxes) == 0:
        return 0.0

    # 1. 按分数降序排列预测框
    pred_boxes = sorted(pred_boxes, key=lambda x: x['score'], reverse=True)

    nd = len(pred_boxes)
    tp = []  # 动态列表
    fp = []

    # 记录哪些 GT 已经被匹配了 (防止重复匹配)
    gt_matched = set()

    for pred in pred_boxes:
        # 寻找匹配的 GT (在所有 GT 中找，而不仅仅是 target)
        # 优化：只筛选同 Sample 的 GT
        candidates = [idx for idx, gt in enumerate(gt_boxes_all) if gt['token'] == pred['token']]

        best_dist = float('inf')
        best_gt_idx = -1

        for gt_idx in candidates:
            if gt_idx in gt_matched:
                continue
            # 计算 2D 中心距离 (nuScenes 标准)
            dist = np.linalg.norm(pred['loc'] - gt_boxes_all[gt_idx]['loc'])
            if dist < best_dist:
                best_dist = dist
                best_gt_idx = gt_idx

        # 判定逻辑
        if best_dist < match_dist_thresh:
            # 匹配到了某个 GT
            matched_gt = gt_boxes_all[best_gt_idx]
            gt_matched.add(best_gt_idx)

            # 关键修正：检查匹配到的 GT 是否属于当前我们要评估的子集 (Target)
            if matched_gt['ann_token'] in target_tokens:
                tp.append(1.0)
                fp.append(0.0)
            else:
                # 匹配到了 GT，但不是当前遮挡等级的 GT (例如评估 Vis=1 时匹配到了 Vis=4 的车)
                # 这种情况应该忽略 (Ignore)，既不算 TP 也不算 FP
                pass
        else:
            # 没有匹配到任何 GT -> 纯粹的误检
            tp.append(0.0)
            fp.append(1.0)

    # 如果所有预测都被忽略了
    if len(tp) == 0:
        return 0.0

    tp = np.array(tp)
    fp = np.array(fp)

    # 3. 计算 AP (使用 NuScenes 标准的 101点插值法 或 VOC 2010+ 全点法)
    # 这里使用类似 VOC 2010+ 的全点法，更平滑
    acc_tp = np.cumsum(tp)
    acc_fp = np.cumsum(fp)
    eps = 1e-6

    # 分母是 target_tokens 的数量，不是 gt_boxes_all 的数量
    total_positives = len(target_tokens)

    rec = acc_tp / total_positives
    prec = acc_tp / (acc_tp + acc_fp + eps)

    ap = 0.0
    # 补点以计算 AUC
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # 使得 Precision 曲线单调递减
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 寻找 Recall 变化点
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


# ================= 2. 数据处理与热力图 =================

def generate_map_heatmap_data(nusc, result_path, dist_bins, score_thresh=0.1):
    with open(result_path, 'r') as f:
        predictions = json.load(f)['results']

    vis_levels = [1, 2, 3, 4]  # 1: 0-40%, 2: 40-60%, 3: 60-80%, 4: 80-100%
    rows = len(vis_levels)
    cols = len(dist_bins) - 1

    # 初始化数据桶
    # gts_all_bucket: 存储某距离段内 *所有* 可见性等级的 GT (用于过滤 FP)
    gts_all_bucket = [{} for _ in range(cols)]
    # gts_target_tokens: 存储具体 (Row, Col) 需要评估的 GT Token 集合
    gts_target_tokens = [[{} for _ in range(cols)] for _ in range(rows)]

    preds_bucket = [{} for _ in range(cols)]

    target_classes = [
        'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'pedestrian', 'motorcycle', 'bicycle', 'barrier', 'traffic_cone'
    ]
    name_map = {
        'vehicle.car': 'car', 'vehicle.truck': 'truck', 'vehicle.bus.rigid': 'bus',
        'vehicle.bus.bendy': 'bus', 'vehicle.trailer': 'trailer',
        'vehicle.construction': 'construction_vehicle', 'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian', 'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian', 'vehicle.motorcycle': 'motorcycle',
        'vehicle.bicycle': 'bicycle', 'static_object.bicycle_rack': 'bicycle',
        'static_object.barrier': 'barrier', 'static_object.traffic_cone': 'traffic_cone'
    }

    print(f"Collecting data from {result_path} ...")

    # --- 数据收集 ---
    # 遍历所有 Samples
    for sample_token in predictions.keys():
        try:
            sample = nusc.get('sample', sample_token)
        except:
            continue

        sd_token = sample['data']['LIDAR_TOP']
        sd_record = nusc.get('sample_data', sd_token)
        ego_pose = nusc.get('ego_pose', sd_record['ego_pose_token'])
        ego_trans = np.array(ego_pose['translation'])

        # 1. 处理 Predictions (按距离分桶，不分可见性)
        raw_preds = predictions[sample_token]
        for p in raw_preds:
            if p['detection_score'] < score_thresh: continue
            p_class = p['detection_name']
            if p_class not in target_classes: continue

            # NuScenes 预测结果是全局坐标
            loc = np.array(p['translation'])
            # 计算到 Ego 的距离 (2D)
            dist = np.linalg.norm(loc[:2] - ego_trans[:2])

            col_idx = -1
            for i in range(len(dist_bins) - 1):
                if dist_bins[i] <= dist < dist_bins[i + 1]:
                    col_idx = i
                    break

            if col_idx != -1:
                if p_class not in preds_bucket[col_idx]:
                    preds_bucket[col_idx][p_class] = []
                preds_bucket[col_idx][p_class].append({
                    'loc': loc[:2],
                    'token': sample_token,
                    'score': p['detection_score']
                })

        # 2. 处理 GTs
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            gt_cat_raw = ann['category_name']

            # 映射类别
            gt_class = None
            for k, v in name_map.items():
                if gt_cat_raw.startswith(k):
                    gt_class = v
                    break
            if gt_class not in target_classes: continue

            gt_loc = np.array(ann['translation'])
            gt_dist = np.linalg.norm(gt_loc[:2] - ego_trans[:2])
            vis_token = int(nusc.get('visibility', ann['visibility_token'])['token'])

            col_idx = -1
            for i in range(len(dist_bins) - 1):
                if dist_bins[i] <= gt_dist < dist_bins[i + 1]:
                    col_idx = i
                    break

            if col_idx != -1:
                # A. 存入 "All GTs" (用于后续做 Ignore 判断)
                if gt_class not in gts_all_bucket[col_idx]:
                    gts_all_bucket[col_idx][gt_class] = []

                # 记录 ann_token 以便唯一标识
                gts_all_bucket[col_idx][gt_class].append({
                    'loc': gt_loc[:2],
                    'token': sample_token,
                    'ann_token': ann_token,
                    'vis': vis_token
                })

                # B. 如果可见性符合，存入 "Target Token Set"
                if vis_token in vis_levels:
                    row_idx = vis_levels.index(vis_token)
                    if gt_class not in gts_target_tokens[row_idx][col_idx]:
                        gts_target_tokens[row_idx][col_idx][gt_class] = set()
                    gts_target_tokens[row_idx][col_idx][gt_class].add(ann_token)

    # --- mAP 计算 ---
    print("Calculating mAP with Ignore logic...")
    match_thresholds = [0.5, 1.0, 2.0, 4.0]
    map_matrix = np.zeros((rows, cols))

    for r in range(rows):  # Visibility
        for c in range(cols):  # Distance
            class_aps = []

            for cls in target_classes:
                # 获取当前距离段下所有的 GT 列表 (包含所有 visibility)
                gts_all = gts_all_bucket[c].get(cls, [])

                # 获取当前格子 (Vis, Dist) 下的目标 GT Token
                target_tokens = gts_target_tokens[r][c].get(cls, set())

                # 获取当前距离段下的预测
                preds = preds_bucket[c].get(cls, [])

                if len(target_tokens) > 0:
                    threshold_aps = []
                    for thresh in match_thresholds:
                        # 核心修改：传入 all GTs 和 target set
                        ap = calculate_masked_ap(gts_all, preds, target_tokens, match_dist_thresh=thresh)
                        threshold_aps.append(ap)

                    avg_class_ap = np.mean(threshold_aps)
                    class_aps.append(avg_class_ap)

            if len(class_aps) > 0:
                map_matrix[r, c] = np.mean(class_aps) * 100
            else:
                map_matrix[r, c] = 0.0  # 或者 np.nan

    return map_matrix

def plot_heatmap(data, dist_bins, title, output_path):
    # 1. 调大画布，防止文字被切掉
    plt.figure(figsize=(14, 10))

    x_labels = [f"{dist_bins[i]}-{dist_bins[i + 1]}m" for i in range(len(dist_bins) - 1)]
    y_labels = [
        "60-100%",
        "40-60%",
        "20-40%",
        "0-20% "
    ]

    # 2. 绘制热力图
    # annot_kws 控制格子内的数字字体
    ax = sns.heatmap(data, annot=True, fmt=".1f", cmap="YlOrRd",
                     xticklabels=x_labels, yticklabels=y_labels,
                     vmin=0, vmax=80,
                     annot_kws={"size": 20, "weight": "bold"})

    # 3. 强制设置 Y 轴标签水平显示 (关键修改)
    # rotation=0: 水平
    # va='center': 垂直居中对齐
    ax.set_yticklabels(y_labels, rotation=90, fontsize=16, fontweight='bold', va='center')

    # 4. 强制设置 X 轴标签水平显示 (防止倾斜)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=16, fontweight='bold')

    # 5. 设置坐标轴标题
    plt.xlabel("Distance to Ego", fontsize=20, fontweight='bold', labelpad=20)
    plt.ylabel("Occlusion  Level", fontsize=20, fontweight='bold', labelpad=20)

    # 6. 调整 Colorbar 字体
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)

    # 7. 关键：自动调整布局，防止左侧长文字被切掉
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Heatmap saved to {output_path}")


# ================= 主程序 =================
if __name__ == "__main__":
    # 配置
    nusc_version = 'v1.0-trainval'
    dataroot = 'data/nuscenes'
    path_root = 'work_dirs/pre_class/results_nusc/'

    # res_path = 'work_dirs/pre_class/results_nusc/RCAlign_24e/results_nusc.json'

    # 距离分段
    dist_bins = [0, 10, 20, 30, 40, 50]  # 可以根据需要改成 [0, 30, 50, 80] 等

    # 初始化 nuScenes
    nusc = NuScenes(version=nusc_version, dataroot=dataroot, verbose=True)

    model_names = ['RCAlign_60e', 'CRT-Fusion', 'RCTrans']
    for model_name in model_names:
        print(f'process the {model_name}')

        res_path = os.path.join(path_root, model_name, 'results_nusc.json')
        out_path = os.path.join(path_root, model_name, "distance_occlusion_heatmap_map.png")
        # 运行
        heatmap_ours = generate_map_heatmap_data(nusc, res_path, dist_bins)
        plot_heatmap(heatmap_ours, dist_bins, "mAP", out_path)