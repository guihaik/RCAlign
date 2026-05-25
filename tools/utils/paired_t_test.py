import numpy as np
from scipy import stats


def calculate_paired_ttest(metric_name, baseline_scores, my_scores):
    """
    计算配对T检验并返回p值和平均提升。
    :param metric_name: 指标名称
    :param baseline_scores: Baseline方法的得分列表
    :param my_scores: 你的方法的得分列表
    :return: (t_stat, p_value, mean_diff)
    """
    base = np.array(baseline_scores)
    mine = np.array(my_scores)

    # 如果你的方法得分低于基线，应使用 'less'
    # 如果只是想看是否有差异，用 'two-sided'
    # 这里我们假设你的方法更好，用 'greater'
    alternative_hypothesis = 'greater' if np.mean(mine) > np.mean(base) else 'less'

    t_stat, p_value = stats.ttest_rel(mine, base, alternative=alternative_hypothesis)

    mean_diff = np.mean(mine - base)

    return t_stat, p_value, mean_diff


# ================= 数据录入 (来自你的 LaTeX 表格) =================
# 使用字典来存储所有数据，方便管理
crt_fusion_data = {
    'NDS': [0.554, 0.553, 0.557],
    'mAP': [0.462, 0.461, 0.468],
    'Car': [0.732, 0.734, 0.732],
    'Truck': [0.419, 0.415, 0.418],
    'Bus': [0.517, 0.511, 0.538],
    'Trailer': [0.235, 0.230, 0.240],
    'C.V.': [0.178, 0.168, 0.188],
    'Ped.': [0.537, 0.533, 0.532],
    'M.C.': [0.440, 0.443, 0.460],
    'Bicycle': [0.424, 0.426, 0.438],
    'T.C.': [0.590, 0.592, 0.596],
    'Barrier': [0.446, 0.424, 0.427]
}

rc_align_data = {
    'NDS': [0.592, 0.593, 0.593],
    'mAP': [0.515, 0.517, 0.514],
    'Car': [0.764, 0.767, 0.765],
    'Truck': [0.489, 0.492, 0.481],
    'Bus': [0.501, 0.534, 0.503],
    'Trailer': [0.172, 0.208, 0.214],
    'C.V.': [0.249, 0.215, 0.240],
    'Ped.': [0.582, 0.592, 0.582],
    'M.C.': [0.580, 0.574, 0.570],
    'Bicycle': [0.509, 0.490, 0.510],
    'T.C.': [0.660, 0.672, 0.652],
    'Barrier': [0.638, 0.627, 0.619]
}

# 定义要计算的指标顺序
metrics_order = ['NDS', 'mAP', 'Car', 'Truck', 'Bus', 'Trailer', 'C.V.',
                 'Ped.', 'M.C.', 'Bicycle', 'T.C.', 'Barrier']

# ================= 打印详细结果和最终总结 =================
summary_results = []

print("--- 详细配对 T 检验结果 ---")
for metric in metrics_order:
    base_scores = crt_fusion_data[metric]
    my_scores = rc_align_data[metric]
    t_stat, p_value, mean_diff = calculate_paired_ttest(metric, base_scores, my_scores)
    summary_results.append((metric, mean_diff, p_value, t_stat))
    print(f"指标: {metric:<10} | 平均提升: {mean_diff:+.4f} | P-value: {p_value:.6f}")

print("\n\n--- 结果总结表格 ---")
print("-" * 60)
print(f"{'Metric':<12} | {'Mean Improvement':<18} | {'P-value':<15} | {'Conclusion'}")
print("-" * 60)
for metric, mean_diff, p_value, t_stat in summary_results:
    conclusion = "Significant (p < 0.05)" if p_value < 0.05 else "Not Significant"
    p_str = f"{p_value:.4f}"
    # if p_value < 0.001:
    #     p_str = "< 0.001"

    print(f"{metric:<12} | {mean_diff:+.4f}{' ':<13} | {p_str:<15} | {conclusion}")
print("-" * 60)