import pickle
import numpy as np
import os
from tqdm import tqdm
from pyquaternion import Quaternion  # 需要 nuscenes-devkit 的依赖


def filter_special_cases(info_path, target_categories=['trailer', 'barrier'], min_dist=0):
    print(f"Loading infos from {info_path} ...")
    with open(info_path, 'rb') as f:
        data = pickle.load(f)

    infos = data['infos'] if 'infos' in data else data

    filtered_results = {cat: [] for cat in target_categories}

    # 映射字典
    keyword_map = {
        'trailer': ['trailer'],
        'barrier': ['barrier'],
    }

    print(f"Total samples: {len(infos)}")

    for i, info in enumerate(tqdm(infos)):
        token = info['token']
        gt_names = info['gt_names']
        gt_boxes = info['gt_boxes']
        num_pts = info.get('num_lidar_pts', np.zeros(len(gt_names)))

        # ==========================================================
        # 1. 获取 Lidar -> Ego 的变换矩阵
        # ==========================================================
        # 这里的 gt_boxes 默认是在 Lidar 坐标系下的
        l2e_r = info['lidar2ego_rotation']  # Quaternion list [w, x, y, z]
        l2e_t = info['lidar2ego_translation']  # Translation list [x, y, z]

        # 构建旋转矩阵
        quaternion = Quaternion(l2e_r)

        # 提取 gt_boxes 的中心点 (N, 3) -> (x, y, z)
        # gt_boxes shape: (N, 7) -> [x, y, z, dx, dy, dz, heading]
        if len(gt_boxes) > 0:
            box_centers_lidar = gt_boxes[:, :3]

            # 旋转: dot product
            # PyQuaternion rotate 方法通常针对单点，批量处理可以用矩阵乘法
            # q.rotate(point) 等价于 q * point * q_conjugate
            # 这里为了简单清晰，使用旋转矩阵: R dot P.T
            rotation_matrix = quaternion.rotation_matrix  # (3, 3)

            # 坐标转换公式: P_ego = R * P_lidar + T
            # 转置以进行矩阵乘法: (3, 3) @ (3, N) -> (3, N)
            box_centers_ego = np.dot(rotation_matrix, box_centers_lidar.T).T + np.array(l2e_t)
        else:
            box_centers_ego = np.array([])

        # ==========================================================

        for cat_key in target_categories:
            target_full_names = keyword_map.get(cat_key, [cat_key])
            indices = [idx for idx, name in enumerate(gt_names) if name in target_full_names]

            if len(indices) > 0:
                objects_info = []
                for idx in indices:
                    # 获取转换后的 Ego 坐标中心
                    center_ego = box_centers_ego[idx]

                    # 原始 box 数据 (保留用于可视化，通常可视化脚本需要知道它是 Lidar 坐标还是 Ego)
                    # 这里我们只存转换后的距离用于筛选
                    raw_box = gt_boxes[idx]

                    # 2. 计算 Ego 距离
                    dist_ego = np.linalg.norm(center_ego[:2])

                    pts = num_pts[idx] if len(num_pts) > idx else -1

                    if dist_ego >= min_dist:
                        objects_info.append({
                            'dist': dist_ego,  # 这是精准的 Ego Distance
                            'pts': pts,
                            'box_lidar': raw_box,  # 原始 Lidar 框 (画图时可能还是需要基于 Lidar 坐标系画)
                            'center_ego': center_ego,
                            'name': gt_names[idx]
                        })

                if len(objects_info) > 0:
                    objects_info.sort(key=lambda x: x['dist'], reverse=True)
                    filtered_results[cat_key].append({
                        'token': token,
                        'count': len(objects_info),
                        'max_dist': objects_info[0]['dist'],
                        'details': objects_info
                    })

    return filtered_results


# ================= 配置路径 =================
args_root_path = 'data/nuscenes'  # 替换为你的路径
info_path = os.path.join(args_root_path, 'nuscenes2d_temporal_infos_val.pkl')

# 1. 筛选 Trailer (重点关注远处的, e.g., > 30m or 40m)
trailer_cases = filter_special_cases(info_path, target_categories=['trailer'], min_dist=0)

# 2. 筛选 Barrier (重点关注这里的, 距离不限，重点看薄的/密集的)
barrier_cases = filter_special_cases(info_path, target_categories=['barrier'], min_dist=0)

# ================= 打印结果供挑选 =================

print("\n====== Top 10 Far Trailer Cases (Candidate Tokens) ======")
# 按最远距离排序，找出最符合 'Far trailer' 的场景
trailer_cases['trailer'].sort(key=lambda x: x['max_dist'], reverse=True)
for item in trailer_cases['trailer'][:10]:
    print(f"Token: {item['token']}")
    print(f"  Max Dist: {item['max_dist']:.2f}m | Total Trailers: {item['count']}")

print("\n====== Top 10 Barrier Cases (Candidate Tokens) ======")
# Barrier 随便选一些，或者选数量多的场景
barrier_cases['barrier'].sort(key=lambda x: x['count'], reverse=True)
for item in barrier_cases['barrier'][:10]:
    print(f"Token: {item['token']}")
    print(f"  Barriers Count: {item['count']} | Max Dist: {item['max_dist']:.2f}m")

# ================= 将 Token 保存到文件以便可视化脚本读取 =================
target_tokens = {
    'trailer_tokens': [x['token'] for x in trailer_cases['trailer']],
    'barrier_tokens': [x['token'] for x in barrier_cases['barrier']]
}

import json
# get sample token which include barrier or tralier

with open('visualization_candidates.json', 'w') as f:
    json.dump(target_tokens, f, indent=4)

print(f"\nCandidates saved to visualization_candidates.json")