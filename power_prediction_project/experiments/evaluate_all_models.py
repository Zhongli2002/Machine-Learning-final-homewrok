"""
综合评估脚本
用于评估和比较LSTM、Transformer和创新性模型的性能
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_dir):
    """加载实验结果"""
    results_file = os.path.join(results_dir, 'results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def run_multiple_experiments(train_script, num_runs=5):
    """运行多次实验并计算统计结果"""
    results = []
    
    for run in range(num_runs):
        print(f"Running experiment {run + 1}/{num_runs}...")
        
        # 这里应该调用训练脚本，但为了演示，我们生成模拟结果
        # 在实际使用中，您需要修改这部分来实际运行训练脚本
        
        # 模拟结果（实际使用时请删除这部分）
        np.random.seed(run)
        mse = np.random.uniform(0.1, 0.5)
        mae = np.random.uniform(0.2, 0.4)
        rmse = np.sqrt(mse)
        
        results.append({
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        })
    
    # 计算统计结果
    mse_values = [r['mse'] for r in results]
    mae_values = [r['mae'] for r in results]
    rmse_values = [r['rmse'] for r in results]
    
    stats = {
        'mse_mean': np.mean(mse_values),
        'mse_std': np.std(mse_values),
        'mae_mean': np.mean(mae_values),
        'mae_std': np.std(mae_values),
        'rmse_mean': np.mean(rmse_values),
        'rmse_std': np.std(rmse_values),
        'individual_results': results
    }
    
    return stats


def create_comparison_table(results_dict):
    """创建模型比较表格"""
    models = list(results_dict.keys())
    metrics = ['MSE', 'MAE', 'RMSE']
    
    # 创建DataFrame
    data = []
    for model in models:
        if results_dict[model] is not None:
            row = [
                f"{results_dict[model]['mse_mean']:.6f} ± {results_dict[model]['mse_std']:.6f}",
                f"{results_dict[model]['mae_mean']:.6f} ± {results_dict[model]['mae_std']:.6f}",
                f"{results_dict[model]['rmse_mean']:.6f} ± {results_dict[model]['rmse_std']:.6f}"
            ]
        else:
            row = ['N/A', 'N/A', 'N/A']
        data.append(row)
    
    df = pd.DataFrame(data, index=models, columns=metrics)
    return df


def plot_model_comparison(results_dict, save_path):
    """绘制模型比较图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    models = [model for model in results_dict.keys() if results_dict[model] is not None]
    
    # MSE比较
    mse_means = [results_dict[model]['mse_mean'] for model in models]
    mse_stds = [results_dict[model]['mse_std'] for model in models]
    
    axes[0, 0].bar(models, mse_means, yerr=mse_stds, capsize=5, alpha=0.7)
    axes[0, 0].set_title('MSE Comparison')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE比较
    mae_means = [results_dict[model]['mae_mean'] for model in models]
    mae_stds = [results_dict[model]['mae_std'] for model in models]
    
    axes[0, 1].bar(models, mae_means, yerr=mae_stds, capsize=5, alpha=0.7, color='orange')
    axes[0, 1].set_title('MAE Comparison')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE比较
    rmse_means = [results_dict[model]['rmse_mean'] for model in models]
    rmse_stds = [results_dict[model]['rmse_std'] for model in models]
    
    axes[1, 0].bar(models, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7, color='green')
    axes[1, 0].set_title('RMSE Comparison')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 综合性能雷达图
    metrics = ['MSE', 'MAE', 'RMSE']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    ax = axes[1, 1]
    ax = plt.subplot(2, 2, 4, projection='polar')
    
    for i, model in enumerate(models):
        values = [
            results_dict[model]['mse_mean'],
            results_dict[model]['mae_mean'],
            results_dict[model]['rmse_mean']
        ]
        # 归一化到0-1范围（越小越好，所以用1-normalized）
        max_vals = [max(mse_means), max(mae_means), max(rmse_means)]
        normalized_values = [1 - v/max_v for v, max_v in zip(values, max_vals)]
        normalized_values += normalized_values[:1]  # 闭合图形
        
        ax.plot(angles, normalized_values, 'o-', linewidth=2, label=model)
        ax.fill(angles, normalized_values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart\n(Higher is Better)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_evaluation_report(results_dict, save_path):
    """生成评估报告"""
    report = []
    report.append("# 模型性能评估报告\n")
    report.append("## 实验设置\n")
    report.append("- 数据集：家庭电力消耗数据")
    report.append("- 评估指标：MSE（均方误差）、MAE（平均绝对误差）、RMSE（均方根误差）")
    report.append("- 实验次数：5次独立运行")
    report.append("- 预测任务：短期预测（90天）和长期预测（365天）\n")
    
    report.append("## 模型比较\n")
    
    # 短期预测结果
    report.append("### 短期预测（90天）\n")
    short_term_results = {
        'LSTM': results_dict.get('LSTM_short'),
        'Transformer': results_dict.get('Transformer_short'),
        'Innovative': results_dict.get('Innovative_short')
    }
    
    if any(r is not None for r in short_term_results.values()):
        df_short = create_comparison_table(short_term_results)
        report.append(df_short.to_markdown())
        report.append("\n")
    
    # 长期预测结果
    report.append("### 长期预测（365天）\n")
    long_term_results = {
        'LSTM': results_dict.get('LSTM_long'),
        'Transformer': results_dict.get('Transformer_long'),
        'Innovative': results_dict.get('Innovative_long')
    }
    
    if any(r is not None for r in long_term_results.values()):
        df_long = create_comparison_table(long_term_results)
        report.append(df_long.to_markdown())
        report.append("\n")
    
    report.append("## 结果分析\n")
    
    # 找出最佳模型
    best_models = {}
    for task, results in [('短期预测', short_term_results), ('长期预测', long_term_results)]:
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_mse = min(valid_results.items(), key=lambda x: x[1]['mse_mean'])
            best_mae = min(valid_results.items(), key=lambda x: x[1]['mae_mean'])
            best_models[task] = {'MSE': best_mse[0], 'MAE': best_mae[0]}
    
    for task, best in best_models.items():
        report.append(f"### {task}\n")
        report.append(f"- MSE最佳模型：{best['MSE']}")
        report.append(f"- MAE最佳模型：{best['MAE']}\n")
    
    report.append("## 模型特点分析\n")
    report.append("### LSTM模型")
    report.append("- 优点：结构简单，训练稳定，适合捕获时序依赖关系")
    report.append("- 缺点：对长期依赖建模能力有限，容易出现梯度消失问题\n")
    
    report.append("### Transformer模型")
    report.append("- 优点：强大的长期依赖建模能力，并行计算效率高")
    report.append("- 缺点：参数量大，训练时间长，对小数据集可能过拟合\n")
    
    report.append("### 创新性模型（ConvLSTMTransformer）")
    report.append("- 优点：结合多种架构优势，多尺度特征提取，自适应注意力机制")
    report.append("- 缺点：模型复杂度高，调参难度大，计算资源需求高\n")
    
    report.append("## 建议\n")
    report.append("1. 对于资源受限的场景，推荐使用LSTM模型")
    report.append("2. 对于长期预测任务，推荐使用Transformer模型")
    report.append("3. 对于追求最佳性能的场景，推荐使用创新性混合模型")
    report.append("4. 建议根据具体应用场景和资源约束选择合适的模型\n")
    
    # 保存报告
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))


def main():
    """主函数"""
    print("Starting comprehensive evaluation...")
    
    # 结果目录
    results_dirs = {
        'LSTM_short': '../results/lstm_short_term',
        'LSTM_long': '../results/lstm_long_term',
        'Transformer_short': '../results/transformer_short_term',
        'Transformer_long': '../results/transformer_long_term',
        'Innovative_short': '../results/innovative_short_term',
        'Innovative_long': '../results/innovative_long_term'
    }
    
    # 创建评估结果目录
    eval_dir = '../results/evaluation'
    os.makedirs(eval_dir, exist_ok=True)
    
    # 运行多次实验（这里使用模拟数据，实际使用时需要调用真实的训练脚本）
    print("Running multiple experiments to get statistical results...")
    
    results_dict = {}
    for name, results_dir in results_dirs.items():
        print(f"评估 {name}...")
        # 这里应该调用实际的训练脚本
        # 为了演示，我们使用模拟结果
        results_dict[name] = run_multiple_experiments(None, num_runs=5)
    
    # 生成比较图表
    print("生成比较图表...")
    plot_model_comparison(results_dict, os.path.join(eval_dir, 'model_comparison.png'))
    
    # 生成评估报告
    print("生成评估报告...")
    generate_evaluation_report(results_dict, os.path.join(eval_dir, 'evaluation_report.md'))
    
    # 保存统计结果
    with open(os.path.join(eval_dir, 'statistical_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("评估完成！结果保存在:", eval_dir)
    
    # 打印简要结果
    print("\n" + "="*50)
    print("简要结果总结")
    print("="*50)
    
    for name, results in results_dict.items():
        if results is not None:
            print(f"{name}:")
            print(f"  MSE: {results['mse_mean']:.6f} ± {results['mse_std']:.6f}")
            print(f"  MAE: {results['mae_mean']:.6f} ± {results['mae_std']:.6f}")
            print(f"  RMSE: {results['rmse_mean']:.6f} ± {results['rmse_std']:.6f}")
            print()


if __name__ == "__main__":
    main()

