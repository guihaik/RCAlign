import os
import json
import shutil
from tqdm import tqdm  # 如果没有安装 tqdm，可以注释掉相关行，或运行 pip install tqdm


def organize_images_by_token(json_path, source_img_dir, output_base_dir):
    """
    根据 json 文件中的 token 分类，将 source_img_dir 中的图片复制到 output_base_dir 下的对应子文件夹中。
    """

    # 1. 读取 JSON 文件
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    print(f"Loading tokens from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 2. 遍历 JSON 中的每个类别
    for category_key, tokens in data.items():
        # 清洗类别名称作为文件夹名
        # 假设 key 是 'trailer_tokens'，我们只想用 'trailer' 作为文件夹名
        folder_name = category_key.replace('_tokens', '')

        target_dir = os.path.join(output_base_dir, folder_name)

        # 创建目标文件夹 (如果不存在)
        os.makedirs(target_dir, exist_ok=True)
        print(f"\nProcessing category: '{folder_name}' -> saving to {target_dir}")
        print(f"Total tokens to copy: {len(tokens)}")

        success_count = 0
        missing_count = 0

        # 3. 遍历 Token 并复制图片
        # 使用 tqdm 显示进度条 (可选)
        for token in tqdm(tokens, desc=f"Copying {folder_name}"):
            # 构造源图片路径 (假设图片名为 token.jpg)
            src_filename = f"{token}.jpg"
            src_path = os.path.join(source_img_dir, src_filename)

            # 构造目标图片路径
            dst_path = os.path.join(target_dir, src_filename)

            if os.path.exists(src_path):
                try:
                    shutil.copy2(src_path, dst_path)  # copy2 保留文件元数据
                    success_count += 1
                except Exception as e:
                    print(f"Failed to copy {src_filename}: {e}")
            else:
                missing_count += 1
                # 如果你想知道哪些图片缺失，可以取消下面这行的注释
                # print(f"Warning: Image not found for token {token}")

        print(f"Done. Success: {success_count}, Missing: {missing_count}")

    print(f"\nAll operations completed. Check your output folder: {output_base_dir}")


# ================= 配置区域 =================

if __name__ == "__main__":
    # 1. 之前生成的 JSON 文件路径
    json_file_path = 'visualization_candidates.json'

    # 2. 存放原始可视化图片的文件夹 (里面全是 token.jpg)
    # 例如: 'work_dirs/vis_results/'
    source_images_path = '/opt/data/private/klh/Object_Detection/StreamPETR_o_r_cha/pic/result_vis_val/'

    # 3. 你想把分类后的图片存到哪里
    # 例如: 'qualitative_analysis_results/'
    output_directory = '/opt/data/private/klh/Object_Detection/StreamPETR_o_r_cha/pic/class'

    # ================= 运行 =================
    organize_images_by_token(json_file_path, source_images_path, output_directory)