"""
L3 贝叶斯控制层可视化分析脚本

支持三种可视化模式:
1. scatter3d: 3D散点图，展示 (mu_H, sigma_H2, k_H) 空间分布
2. trajectory: 时序轨迹图，展示多条件下熵值变化
3. classify: 分类准确率曲线，评估信号可区分性

使用示例:
    # 3D散点图
    python tests/l3_visualization.py --mode scatter3d --data results/l3_trace.csv

    # 时序轨迹
    python tests/l3_visualization.py --mode trajectory --data results/l3_trace.csv

    # 分类评估
    python tests/l3_visualization.py --mode classify --data results/l3_trace.csv
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


COLORS = {
    'normal': '#2ecc71',
    'noise_injection': '#e74c3c',
    'bias_injection': '#3498db'
}

LABELS = {
    'normal': '正常条件 (H1)',
    'noise_injection': '噪声注入 (H2)',
    'bias_injection': '偏见注入 (H3)'
}


def load_csv(filepath: str) -> List[Dict]:
    """加载CSV数据"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                'timestamp': row['timestamp'],
                'condition': row['condition'],
                'session_id': row['session_id'],
                'turn': int(row['turn']),
                'mu_H': float(row['mu_H']),
                'sigma_H2': float(row['sigma_H2']),
                'k_H': float(row['k_H']),
                'p_harm_raw': float(row['p_harm_raw']),
            })
    return records


def plot_scatter3d(records: List[Dict], output_path: str):
    """绘制3D散点图"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Error: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    data_by_condition = defaultdict(list)
    for r in records:
        data_by_condition[r['condition']].append(r)

    for condition, color in COLORS.items():
        if condition not in data_by_condition:
            continue

        data = data_by_condition[condition]
        xs = [r['mu_H'] for r in data]
        ys = [r['sigma_H2'] for r in data]
        zs = [r['k_H'] for r in data]

        ax.scatter(xs, ys, zs, c=color, label=LABELS[condition], alpha=0.7, s=50)

    ax.set_xlabel('mu_H (后验熵均值)', fontsize=10, labelpad=10)
    ax.set_ylabel('sigma_H2 (后验熵方差)', fontsize=10, labelpad=10)
    ax.set_zlabel('k_H (熵变化趋势)', fontsize=10, labelpad=10)
    ax.set_title('L3 Trace: 3D信号空间分布', fontsize=14, fontweight='bold')

    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"3D散点图已保存: {output_path}")
    plt.close()


def plot_trajectory(records: List[Dict], output_path: str):
    """绘制时序轨迹图"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for visualization")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    data_by_condition = defaultdict(list)
    for r in records:
        data_by_condition[r['condition']].append(r)

    for condition, color in COLORS.items():
        if condition not in data_by_condition:
            continue

        data = data_by_condition[condition]

        turns = [r['turn'] for r in data]
        mu_H = [r['mu_H'] for r in data]
        sigma_H2 = [r['sigma_H2'] for r in data]

        axes[0].plot(turns, mu_H, 'o-', color=color, label=LABELS[condition], alpha=0.7)
        axes[1].plot(turns, sigma_H2, 's-', color=color, label=LABELS[condition], alpha=0.7)

    axes[0].set_xlabel('对话轮次 (Turn)', fontsize=10)
    axes[0].set_ylabel('mu_H (后验熵均值)', fontsize=10)
    axes[0].set_title('mu_H 时序轨迹', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('对话轮次 (Turn)', fontsize=10)
    axes[1].set_ylabel('sigma_H2 (后验熵方差)', fontsize=10)
    axes[1].set_title('sigma_H2 时序轨迹', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"时序轨迹图已保存: {output_path}")
    plt.close()


def calculate_classification_accuracy(
    records: List[Dict],
    window_sizes: List[int]
) -> Dict[int, Dict[str, float]]:
    """
    计算滑动窗口分类准确率

    使用简单规则进行3类分类:
    - H1 (正常): mu_H > 0.4 且 sigma_H2 < 0.15
    - H2 (噪声): sigma_H2 > 0.15
    - H3 (偏见): mu_H < 0.4 且 p_harm_raw > 0.3
    """
    results = {}

    for L in window_sizes:
        correct = 0
        total = 0

        for i in range(L, len(records)):
            window = records[i-L:i]

            mu_H_mean = np.mean([r['mu_H'] for r in window])
            sigma_H2_mean = np.mean([r['sigma_H2'] for r in window])
            p_harm_mean = np.mean([r['p_harm_raw'] for r in window])

            if sigma_H2_mean > 0.15:
                predicted = 'noise_injection'
            elif mu_H_mean < 0.4 and p_harm_mean > 0.3:
                predicted = 'bias_injection'
            else:
                predicted = 'normal'

            actual = records[i]['condition']
            if predicted == actual:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        results[L] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }

    return results


def plot_classification(records: List[Dict], output_path: str):
    """绘制分类准确率曲线"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for visualization")
        return

    window_sizes = [1, 3, 5, 10, 15, 20]
    results = calculate_classification_accuracy(records, window_sizes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    accuracies = [results[L]['accuracy'] * 100 for L in window_sizes]

    axes[0].bar(range(len(window_sizes)), accuracies, color='#3498db', alpha=0.8)
    axes[0].set_xticks(range(len(window_sizes)))
    axes[0].set_xticklabels([f'L={L}' for L in window_sizes])
    axes[0].set_xlabel('滑动窗口长度', fontsize=10)
    axes[0].set_ylabel('分类准确率 (%)', fontsize=10)
    axes[0].set_title('3类分类准确率 vs 窗口长度', fontsize=12, fontweight='bold')
    axes[0].set_ylim([0, 100])
    axes[0].axhline(y=85, color='red', linestyle='--', label='85% 基线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    for i, (L, acc) in enumerate(zip(window_sizes, accuracies)):
        axes[0].text(i, acc + 2, f'{acc:.1f}%', ha='center', fontsize=9)

    axes[1].plot(window_sizes, accuracies, 'o-', color='#3498db', linewidth=2, markersize=8)
    axes[1].set_xlabel('滑动窗口长度', fontsize=10)
    axes[1].set_ylabel('分类准确率 (%)', fontsize=10)
    axes[1].set_title('分类准确率趋势', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 100])
    axes[1].axhline(y=85, color='red', linestyle='--', label='85% 基线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"分类评估图已保存: {output_path}")

    print("\n分类评估结果:")
    print("-" * 50)
    print(f"{'窗口L':<10} {'准确率':<12} {'正确/总数':<15}")
    print("-" * 50)
    for L in window_sizes:
        r = results[L]
        print(f"L={L:<6} {r['accuracy']*100:>6.1f}%    {r['correct']}/{r['total']}")

    L10_acc = results[10]['accuracy'] * 100 if 10 in results else 0
    if L10_acc > 85:
        print(f"\n✅ L=10 分类准确率 {L10_acc:.1f}% > 85% 基线")
        print("   信号可区分性验证通过！")
    else:
        print(f"\n⚠️ L=10 分类准确率 {L10_acc:.1f}% <= 85% 基线")
        print("   需要更多数据或调整分类规则")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='L3 Trace 可视化分析')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['scatter3d', 'trajectory', 'classify'],
                       help='可视化模式')
    parser.add_argument('--data', type=str, required=True,
                       help='CSV数据文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图片路径')

    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"Error: 数据文件不存在: {args.data}")
        return

    records = load_csv(args.data)
    print(f"已加载 {len(records)} 条记录")

    if not records:
        print("Error: 数据为空")
        return

    conditions = set(r['condition'] for r in records)
    condition_counts = {c: sum(1 for r in records if r['condition'] == c) for c in conditions}
    print(f"条件分布: {condition_counts}")

    output_dir = Path(args.data).parent / 'visualization'
    output_dir.mkdir(exist_ok=True)

    if args.mode == 'scatter3d':
        output_path = args.output or str(output_dir / 'scatter3d.png')
        plot_scatter3d(records, output_path)

    elif args.mode == 'trajectory':
        output_path = args.output or str(output_dir / 'trajectory.png')
        plot_trajectory(records, output_path)

    elif args.mode == 'classify':
        output_path = args.output or str(output_dir / 'classification.png')
        plot_classification(records, output_path)


if __name__ == '__main__':
    main()
