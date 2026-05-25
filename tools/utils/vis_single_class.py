import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes


def get_samples_with_trailer_in_val(nusc):
    """
    获取验证集(val)中所有包含 trailer 的 sample_token
    """
    print("Filtering samples with trailers in 'val' split...")

    # 1. 获取 val 集的所有场景名称
    splits = create_splits_scenes()
    val_scenes = set(splits['val'])

    trailer_sample_tokens = set()

    # 2. 遍历所有注释，筛选类别 + 验证集场景
    for ann in nusc.sample_annotation:
        if 'movable_object.barrier' in ann['category_name']: # vehicle.trailer
            sample = nusc.get('sample', ann['sample_token'])
            scene = nusc.get('scene', sample['scene_token'])

            # 检查是否属于验证集
            if scene['name'] in val_scenes:
                trailer_sample_tokens.add(sample['token'])

    results = list(trailer_sample_tokens)
    print(f"Found {len(results)} unique samples with trailers in validation set.")
    return results


def render_filtered_camera(nusc, sensor_token, ax, target_keyword='trailer'):
    """
    在指定 ax 上绘制相机图像，且只绘制 trailer 的框
    """
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(sensor_token)

    # 显示图片
    im = Image.open(data_path)
    ax.imshow(im)

    # 设置标题
    sd_record = nusc.get('sample_data', sensor_token)
    ax.set_title(sd_record['channel'].replace('CAM_', ''), fontsize=10)
    ax.axis('off')

    # 筛选并绘制框 (红色)
    for box in boxes:
        if target_keyword in box.name:
            if box.center[2] > 0:  # 只画相机前方的
                box.render(ax, view=camera_intrinsic, normalize=True, colors=('r', 'r', 'r'), linewidth=2)


def render_filtered_lidar(nusc, sensor_token, ax, target_keyword='trailer'):
    """
    在指定 ax 上绘制 BEV 点云，且只绘制 trailer 的框
    """
    data_path, boxes, _ = nusc.get_sample_data(sensor_token)
    pc = LidarPointCloud.from_file(data_path)

    # 绘制点云 (灰色, 降采样)
    ax.scatter(pc.points[0, ::5], pc.points[1, ::5], c='#404040', s=0.5, alpha=0.8)

    # 设置 BEV 范围
    limit = 60
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_title("LIDAR BEV (Trailers Only)")
    ax.axis('off')
    ax.set_aspect('equal')

    # 筛选并绘制框 (红色, BEV模式)
    for box in boxes:
        if target_keyword in box.name:
            # 这里的 view=np.eye(3) 只是为了让 render 函数能运行，实际在BEV下主要依赖xy坐标
            box.render(ax, view=np.eye(3), colors=('r', 'r', 'r'), linewidth=2)


def visualize_sample_and_save(nusc, sample_token, output_dir):
    sample = nusc.get('sample', sample_token)

    # 创建画布 (24x12 英寸)
    # 布局: 2行4列 (前3列相机，第4列LiDAR)
    fig = plt.figure(figsize=(24, 12))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1.5])

    # === 左侧：相机图像 (2x3) ===
    cam_order = [
        ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
        ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    ]

    for row in range(2):
        for col in range(3):
            sensor_name = cam_order[row][col]
            ax = fig.add_subplot(gs[row, col])

            if sensor_name in sample['data']:
                render_filtered_camera(nusc, sample['data'][sensor_name], ax, target_keyword='barrier')
            else:
                ax.text(0.5, 0.5, "No Data", ha='center')

    # === 右侧：LiDAR BEV (跨两行) ===
    ax_lidar = fig.add_subplot(gs[:, 3])
    if 'LIDAR_TOP' in sample['data']:
        render_filtered_lidar(nusc, sample['data']['LIDAR_TOP'], ax_lidar, target_keyword='barrier')

    plt.tight_layout()

    # === 保存逻辑 ===
    file_name = f"{sample_token}.jpg"
    save_path = os.path.join(output_dir, file_name)

    plt.savefig(save_path, dpi=150)  # dpi=150 保证清晰度但文件不会过大
    plt.close(fig)  # 关键：关闭图像释放内存，否则循环多了会爆内存
    print(f"Saved: {save_path}")


# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 配置路径
    #vis the gt single class in the image and bev
    dataroot = 'data/nuscenes'  # 请修改为你的 nuScenes 路径
    # output_directory = 'pic/vis_trailer_results'  # 结果保存路径
    output_directory = 'pic/vis_barrier_results'  # 结果保存路径

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 2. 初始化
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)

    # 3. 获取 val 集中的 trailer 样本
    target_samples = get_samples_with_trailer_in_val(nusc)

    # 4. 批量处理并保存
    # 这里处理前 20 个作为示例，如果想处理全部，去掉 [:20]
    total = len(target_samples)
    print(f"Start generating images for {total} samples...")

    for i, token in enumerate(target_samples):
        print(f"Processing {i + 1}/{total}: {token}")
        try:
            visualize_sample_and_save(nusc, token, output_directory)
        except Exception as e:
            print(f"Error processing {token}: {e}")

    print(f"\nAll done! Check the folder: {output_directory}")