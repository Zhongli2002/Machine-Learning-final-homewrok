#!/usr/bin/env python3
"""
Final Comprehensive Power Prediction Experiment
==============================================

This script runs a comprehensive experiment comparing 7 models (excluding MoE):
- LSTM (standard & large)
- Transformer (standard & large)  
- Informer (standard & large)
- MoE V2 (standard & large)
- Enhanced LSTM (standard & large)
- Enhanced Transformer (standard & large)
- Innovative Model (standard & large)

Each model is run 5 times for 100 epochs with statistical analysis and prediction curves.
Results are saved to ./results_final directory.
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from typing import Dict, List, Tuple
import warnings
import logging
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rizhi.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinalComprehensiveExperiment:
    
    def __init__(self, base_epochs: int = 100, num_runs: int = 5, test_mode: bool = False):
        self.base_epochs = base_epochs
        self.num_runs = num_runs
        self.test_mode = test_mode
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if test_mode:
            self.results_dir = f"results_test/test_experiment_{self.timestamp}"
            # Test mode: all 7 models, 1 task, 1 size
            self.models = {
                'lstm': {'script': 'train_lstm.py', 'tasks': ['short'], 'name': 'LSTM'},
                'transformer': {'script': 'train_transformer.py', 'tasks': ['short'], 'name': 'Transformer'},
                'informer': {'script': 'train_informer.py', 'tasks': ['short'], 'name': 'Informer'},
                'moe_v2': {'script': 'train_moe_v2.py', 'tasks': ['short'], 'name': 'MoE V2'},
                'enhanced_lstm': {'script': 'train_enhanced_lstm.py', 'tasks': ['short'], 'name': 'Enhanced LSTM'},
                'enhanced_transformer': {'script': 'train_enhanced_transformer.py', 'tasks': ['short'], 'name': 'Enhanced Transformer'},
                'innovative': {'script': 'train_innovative.py', 'tasks': ['short'], 'name': 'Innovative Model'}
            }
            self.model_sizes = ['standard']
        else:
            self.results_dir = f"results_final/final_experiment_{self.timestamp}"
            # Full mode: 7 models (excluding MoE)
            self.models = {
                'lstm': {'script': 'train_lstm.py', 'tasks': ['short', 'long'], 'name': 'LSTM'},
                'transformer': {'script': 'train_transformer.py', 'tasks': ['short', 'long'], 'name': 'Transformer'},
                'informer': {'script': 'train_informer.py', 'tasks': ['short', 'long'], 'name': 'Informer'},
                'moe_v2': {'script': 'train_moe_v2.py', 'tasks': ['short', 'long'], 'name': 'MoE V2'},
                'enhanced_lstm': {'script': 'train_enhanced_lstm.py', 'tasks': ['short', 'long'], 'name': 'Enhanced LSTM'},
                'enhanced_transformer': {'script': 'train_enhanced_transformer.py', 'tasks': ['short', 'long'], 'name': 'Enhanced Transformer'},
                'innovative': {'script': 'train_innovative.py', 'tasks': ['short', 'long'], 'name': 'Innovative Model'}
            }
            self.model_sizes = ['standard', 'large']
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Results storage
        self.all_results = {}
        self.failed_experiments = []
        
        logger.info(f"Final Comprehensive Experiment initialized")
        logger.info(f"Results will be saved to: {self.results_dir}")
        logger.info(f"Configuration: {num_runs} runs × {base_epochs} epochs per experiment")
        logger.info(f"Total experiments: {len(self.models) * len(self.model_sizes) * 2 * num_runs}")
        logger.info(f"Models included: {', '.join([model['name'] for model in self.models.values()])}")
        
    def run_single_experiment(self, model_name: str, task: str, model_size: str, run_id: int) -> Dict:
        """Run a single experiment and return results."""
        
        experiment_id = f"{model_name}_{task}_{model_size}_run{run_id:02d}"
        logger.info(f"Running: {experiment_id}")
        
        # Create experiment-specific results directory
        exp_results_dir = os.path.join(self.results_dir, experiment_id)
        os.makedirs(exp_results_dir, exist_ok=True)
        
        # Build command (removed GPU parameters as they're not supported)
        script_path = f"experiments/{self.models[model_name]['script']}"
        
        # Handle different parameter formats for different models
        if model_name in ['enhanced_lstm', 'enhanced_transformer']:
            # These models use --task_type and expect 'short_term'/'long_term'
            task_param = f"{task}_term"
            cmd = [
                sys.executable, script_path,
                "--task_type", task_param,
                "--epochs", str(self.base_epochs),
                "--model_size", model_size,
                "--results_dir", exp_results_dir
            ]
        else:
            # Standard models use --task
            cmd = [
                sys.executable, script_path,
                "--task", task,
                "--epochs", str(self.base_epochs),
                "--model_size", model_size,
                "--results_dir", exp_results_dir
            ]
        
        # Run experiment
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)  # 40 minutes timeout
            
            if result.returncode == 0:
                # Load results
                results_file = os.path.join(exp_results_dir, 'results.json')
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        metrics = json.load(f)
                    
                    experiment_result = {
                        'experiment_id': experiment_id,
                        'model_name': model_name,
                        'model_display_name': self.models[model_name]['name'],
                        'task': task,
                        'model_size': model_size,
                        'run_id': run_id,
                        'mse': metrics.get('mse', float('inf')),
                        'mae': metrics.get('mae', float('inf')),
                        'rmse': metrics.get('rmse', float('inf')),
                        'runtime': time.time() - start_time,
                        'status': 'success',
                        'results_dir': exp_results_dir
                    }
                    
                    # Add MoE-specific metrics if available
                    if 'lstm_weight' in metrics:
                        experiment_result['lstm_weight'] = metrics['lstm_weight']
                        experiment_result['transformer_weight'] = metrics['transformer_weight']
                    
                    # Load predictions and ground truth for plotting
                    predictions_file = os.path.join(exp_results_dir, 'predictions.npz')
                    if os.path.exists(predictions_file):
                        pred_data = np.load(predictions_file)
                        experiment_result['predictions'] = pred_data['predictions'].tolist()
                        experiment_result['ground_truth'] = pred_data['ground_truth'].tolist()
                    
                    logger.info(f"SUCCESS: {experiment_id} - MSE={metrics.get('mse', 'N/A'):.6f}, "
                              f"MAE={metrics.get('mae', 'N/A'):.6f}, Runtime={experiment_result['runtime']:.1f}s")
                    
                    return experiment_result
                else:
                    logger.error(f"FAILED: {experiment_id} - Results file not found")
                    return self._create_failed_result(experiment_id, model_name, task, model_size, run_id, 
                                                    "Results file not found", time.time() - start_time)
            else:
                logger.error(f"FAILED: {experiment_id} - Return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return self._create_failed_result(experiment_id, model_name, task, model_size, run_id,
                                                f"Process failed: {result.stderr}", time.time() - start_time)
                
        except subprocess.TimeoutExpired:
            logger.error(f"FAILED: {experiment_id} - Timeout after 40 minutes")
            return self._create_failed_result(experiment_id, model_name, task, model_size, run_id,
                                            "Timeout", time.time() - start_time)
        except Exception as e:
            logger.error(f"FAILED: {experiment_id} - Exception: {str(e)}")
            return self._create_failed_result(experiment_id, model_name, task, model_size, run_id,
                                            f"Exception: {str(e)}", time.time() - start_time)
    
    def _create_failed_result(self, experiment_id: str, model_name: str, task: str, 
                            model_size: str, run_id: int, error: str, runtime: float) -> Dict:
        """Create a failed experiment result."""
        return {
            'experiment_id': experiment_id,
            'model_name': model_name,
            'model_display_name': self.models[model_name]['name'],
            'task': task,
            'model_size': model_size,
            'run_id': run_id,
            'mse': float('inf'),
            'mae': float('inf'),
            'rmse': float('inf'),
            'runtime': runtime,
            'status': 'failed',
            'error': error,
            'results_dir': None
        }
    
    def run_all_experiments(self):
        """Run all experiments with dual GPU optimization."""
        logger.info("Starting Final Comprehensive Experiment")
        
        total_experiments = len(self.models) * len(self.model_sizes) * 2 * self.num_runs
        completed = 0
        
        for model_name in self.models:
            for model_size in self.model_sizes:
                for task in self.models[model_name]['tasks']:
                    
                    # Initialize results for this configuration
                    config_key = f"{model_name}_{task}_{model_size}"
                    self.all_results[config_key] = []
                    
                    logger.info(f"Running {self.models[model_name]['name']} {model_size} {task} task ({self.num_runs} runs)")
                    
                    for run_id in range(self.num_runs):
                        result = self.run_single_experiment(model_name, task, model_size, run_id)
                        self.all_results[config_key].append(result)
                        
                        if result['status'] == 'failed':
                            self.failed_experiments.append(result)
                        
                        completed += 1
                        progress = (completed / total_experiments) * 100
                        logger.info(f"Progress: {completed}/{total_experiments} ({progress:.1f}%)")
                        
                        # Save intermediate results
                        self._save_intermediate_results()
        
        logger.info(f"All experiments completed!")
        logger.info(f"Total: {completed}/{total_experiments}")
        logger.info(f"Failed: {len(self.failed_experiments)}")
        
    def _save_intermediate_results(self):
        """Save intermediate results to prevent data loss."""
        results_file = os.path.join(self.results_dir, 'intermediate_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'all_results': self.all_results,
                'failed_experiments': self.failed_experiments,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
    
    def analyze_results(self):
        """Perform comprehensive statistical analysis."""
        logger.info("Analyzing Results...")
        
        # Create analysis directory
        analysis_dir = os.path.join(self.results_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Prepare data for analysis
        analysis_data = []
        
        for config_key, results in self.all_results.items():
            successful_results = [r for r in results if r['status'] == 'success']
            
            if len(successful_results) > 0:
                mse_values = [r['mse'] for r in successful_results]
                mae_values = [r['mae'] for r in successful_results]
                rmse_values = [r['rmse'] for r in successful_results]
                runtimes = [r['runtime'] for r in successful_results]
                
                # Calculate statistics
                stats_dict = {
                    'config': config_key,
                    'model_name': successful_results[0]['model_name'],
                    'model_display_name': successful_results[0]['model_display_name'],
                    'task': successful_results[0]['task'],
                    'model_size': successful_results[0]['model_size'],
                    'successful_runs': len(successful_results),
                    'total_runs': len(results),
                    'success_rate': len(successful_results) / len(results),
                    
                    # MSE statistics
                    'mse_mean': np.mean(mse_values),
                    'mse_std': np.std(mse_values),
                    'mse_min': np.min(mse_values),
                    'mse_max': np.max(mse_values),
                    'mse_median': np.median(mse_values),
                    
                    # MAE statistics
                    'mae_mean': np.mean(mae_values),
                    'mae_std': np.std(mae_values),
                    'mae_min': np.min(mae_values),
                    'mae_max': np.max(mae_values),
                    'mae_median': np.median(mae_values),
                    
                    # RMSE statistics
                    'rmse_mean': np.mean(rmse_values),
                    'rmse_std': np.std(rmse_values),
                    'rmse_min': np.min(rmse_values),
                    'rmse_max': np.max(rmse_values),
                    'rmse_median': np.median(rmse_values),
                    
                    # Runtime statistics
                    'runtime_mean': np.mean(runtimes),
                    'runtime_std': np.std(runtimes),
                    'runtime_min': np.min(runtimes),
                    'runtime_max': np.max(runtimes),
                }
                
                analysis_data.append(stats_dict)
        
        # Create DataFrame
        df = pd.DataFrame(analysis_data)
        
        # Save detailed statistics
        df.to_csv(os.path.join(analysis_dir, 'detailed_statistics.csv'), index=False)
        
        # Generate summary statistics in required format
        self._generate_summary_statistics(df, analysis_dir)
        
        # Generate visualizations
        self._generate_visualizations(df, analysis_dir)
        
        # Generate prediction comparison plots
        self._generate_prediction_plots(analysis_dir)
        
        # Statistical significance testing
        self._perform_statistical_tests(df, analysis_dir)
        
        # Generate final report
        self._generate_final_report(df, analysis_dir)
        
        logger.info(f"Analysis completed! Results saved to: {analysis_dir}")
    
    def _generate_summary_statistics(self, df: pd.DataFrame, analysis_dir: str):
        """Generate summary statistics in the required format."""
        logger.info("Generating summary statistics...")
        
        # Create summary table with mean and std for MSE and MAE
        summary_data = []
        
        for _, row in df.iterrows():
            summary_data.append({
                'Model': row['model_display_name'],
                'Task': row['task'].title(),
                'Size': row['model_size'].title(),
                'Runs': row['successful_runs'],
                'MSE_Mean': f"{row['mse_mean']:.2f}",
                'MSE_Std': f"{row['mse_std']:.2f}",
                'MAE_Mean': f"{row['mae_mean']:.2f}",
                'MAE_Std': f"{row['mae_std']:.2f}",
                'RMSE_Mean': f"{row['rmse_mean']:.2f}",
                'RMSE_Std': f"{row['rmse_std']:.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(analysis_dir, 'summary_statistics.csv'), index=False)
        
        logger.info("Summary statistics saved to summary_statistics.csv")
    
    def _generate_prediction_plots(self, analysis_dir: str):
        """Generate power prediction vs ground truth comparison plots."""
        logger.info("Generating prediction comparison plots...")
        
        # Find experiments with prediction data
        prediction_configs = []
        for config, results in self.all_results.items():
            for result in results:
                if 'predictions' in result and 'ground_truth' in result:
                    prediction_configs.append((config, result))
                    break  # Only need one example per config
        
        if not prediction_configs:
            logger.warning("No prediction data found for visualization")
            return
        
        # Create comparison plots
        n_plots = min(len(prediction_configs), 12)  # Limit to 12 plots
        cols = 3
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        fig.suptitle('Power Prediction vs Ground Truth Comparison', fontsize=16, fontweight='bold')
        
        for i, (config, result) in enumerate(prediction_configs[:n_plots]):
            if i >= len(axes):
                break
                
            predictions = np.array(result['predictions'])
            ground_truth = np.array(result['ground_truth'])
            
            # Limit to first 100 points for clarity
            if len(predictions) > 100:
                predictions = predictions[:100]
                ground_truth = ground_truth[:100]
            
            time_steps = range(len(predictions))
            
            axes[i].plot(time_steps, ground_truth, label='Ground Truth', 
                        color='blue', linewidth=2, alpha=0.8)
            axes[i].plot(time_steps, predictions, label='Predictions', 
                        color='red', linewidth=2, alpha=0.8, linestyle='--')
            
            # Calculate correlation
            correlation = np.corrcoef(predictions, ground_truth)[0, 1]
            
            # Format config name
            config_parts = config.split('_')
            if len(config_parts) >= 3:
                model = '_'.join(config_parts[:-2])  # Handle multi-word model names
                task = config_parts[-2]
                size = config_parts[-1]
            else:
                model, task, size = config_parts[0], config_parts[1] if len(config_parts) > 1 else 'unknown', config_parts[2] if len(config_parts) > 2 else 'unknown'
            title = f"{model.replace('_', ' ').title()}\n{task.replace('_', ' ').title()} - {size.title()}"
            title += f"\nCorr: {correlation:.3f}"
            
            axes[i].set_title(title, fontsize=10)
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Power')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        pred_path = os.path.join(analysis_dir, 'prediction_comparisons.png')
        plt.savefig(pred_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Prediction comparison plots saved to: {pred_path}")
        
    def _generate_visualizations(self, df: pd.DataFrame, analysis_dir: str):
        """Generate comprehensive visualizations."""
        logger.info("Generating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Performance comparison by metric
        metrics = ['mse_mean', 'mae_mean', 'rmse_mean']
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Model Performance Comparison (7 Models)', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            # Create pivot table for heatmap
            pivot_data = df.pivot_table(
                values=metric, 
                index=['model_display_name', 'model_size'], 
                columns='task', 
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_data, annot=True, fmt='.6f', cmap='YlOrRd', ax=ax)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel('Task')
            ax.set_ylabel('Model (Size)')
        
        # 4. Runtime comparison
        ax = axes[1, 1]
        pivot_runtime = df.pivot_table(
            values='runtime_mean', 
            index=['model_display_name', 'model_size'], 
            columns='task', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_runtime, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax)
        ax.set_title('Average Runtime (seconds)')
        ax.set_xlabel('Task')
        ax.set_ylabel('Model (Size)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'performance_heatmaps.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Model ranking visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Model Rankings by Performance Metrics', fontsize=16, fontweight='bold')
        
        tasks = ['short', 'long']
        metrics = ['mse_mean', 'mae_mean']
        
        for i, task in enumerate(tasks):
            for j, metric in enumerate(metrics):
                ax = axes[i, j]
                
                task_data = df[df['task'] == task].copy()
                task_data = task_data.sort_values(metric)
                
                bars = ax.barh(range(len(task_data)), task_data[metric])
                ax.set_yticks(range(len(task_data)))
                ax.set_yticklabels([f"{row['model_display_name']}_{row['model_size']}" 
                                  for _, row in task_data.iterrows()])
                ax.set_xlabel(metric.replace('_', ' ').title())
                ax.set_title(f'{task.title()}-term Task')
                
                # Color bars by rank
                colors = plt.cm.get_cmap('RdYlGn_r')(np.linspace(0, 1, len(bars)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'model_rankings.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizations generated successfully!")
        
    def _perform_statistical_tests(self, df: pd.DataFrame, analysis_dir: str):
        """Perform statistical significance tests."""
        logger.info("Performing statistical tests...")
        
        # Prepare data for pairwise comparisons
        test_results = []
        
        # Get all model configurations
        configs = df['config'].unique()
        
        # Perform pairwise t-tests for MSE
        for i, config1 in enumerate(configs):
            for config2 in configs[i+1:]:
                # Get the raw data for both configurations
                results1 = self.all_results[config1]
                results2 = self.all_results[config2]
                
                # Filter successful runs
                mse1 = [r['mse'] for r in results1 if r['status'] == 'success']
                mse2 = [r['mse'] for r in results2 if r['status'] == 'success']
                
                if len(mse1) >= 3 and len(mse2) >= 3:  # Need at least 3 samples for t-test
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(mse1, mse2)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(mse1) - 1) * np.var(mse1) + 
                                        (len(mse2) - 1) * np.var(mse2)) / 
                                       (len(mse1) + len(mse2) - 2))
                    cohens_d = (np.mean(mse1) - np.mean(mse2)) / pooled_std
                    
                    test_results.append({
                        'config1': config1,
                        'config2': config2,
                        'mean_mse1': np.mean(mse1),
                        'mean_mse2': np.mean(mse2),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant': p_value < 0.05,
                        'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
                    })
        
        # Save test results
        test_df = pd.DataFrame(test_results)
        test_df.to_csv(os.path.join(analysis_dir, 'statistical_tests.csv'), index=False)
        
        logger.info(f"Statistical tests completed! {len(test_results)} pairwise comparisons performed.")
        
    def _generate_final_report(self, df: pd.DataFrame, analysis_dir: str):
        """Generate the final comprehensive report."""
        logger.info("Generating final report...")
        
        report_path = os.path.join(analysis_dir, 'FINAL_REPORT.md')
        
        with open(report_path, 'w') as f:
            f.write("# Final Comprehensive Power Prediction Experiment Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Experiment Configuration:**\n")
            f.write(f"- Runs per model: {self.num_runs}\n")
            f.write(f"- Epochs per run: {self.base_epochs}\n")
            f.write(f"- Total experiments: {len(self.models) * len(self.model_sizes) * 2 * self.num_runs}\n")
            f.write(f"- Models included: {', '.join([model['name'] for model in self.models.values()])}\n")
            f.write(f"- Dataset: train.csv and test.csv\n")
            f.write(f"- Evaluation metrics: MSE and MAE\n")
            f.write(f"- GPU utilization: Dual GPU (alternating)\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Overall success rate
            total_experiments = sum(len(results) for results in self.all_results.values())
            successful_experiments = sum(len([r for r in results if r['status'] == 'success']) 
                                       for results in self.all_results.values())
            success_rate = successful_experiments / total_experiments * 100
            
            f.write(f"- **Overall Success Rate:** {success_rate:.1f}% ({successful_experiments}/{total_experiments})\n")
            f.write(f"- **Failed Experiments:** {len(self.failed_experiments)}\n")
            f.write(f"- **Models Tested:** {len(self.models)} (MoE excluded as requested)\n\n")
            
            # Best performing models
            for task in ['short', 'long']:
                task_data = df[df['task'] == task].copy()
                if len(task_data) > 0:
                    best_mse = task_data.loc[task_data['mse_mean'].idxmin()]
                    best_mae = task_data.loc[task_data['mae_mean'].idxmin()]
                    
                    f.write(f"### {task.title()}-term Prediction Champions\n\n")
                    f.write(f"**Best MSE:** {best_mse['model_display_name']} {best_mse['model_size']} "
                           f"(MSE: {best_mse['mse_mean']:.6f} ± {best_mse['mse_std']:.6f})\n\n")
                    f.write(f"**Best MAE:** {best_mae['model_display_name']} {best_mae['model_size']} "
                           f"(MAE: {best_mae['mae_mean']:.6f} ± {best_mae['mae_std']:.6f})\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write("### Performance Summary Table\n\n")
            
            # Create summary table
            f.write("| Model | Size | Task | MSE (mean ± std) | MAE (mean ± std) | RMSE (mean ± std) | Success Rate |\n")
            f.write("|-------|------|------|------------------|------------------|-------------------|-------------|\n")
            
            for _, row in df.iterrows():
                f.write(f"| {row['model_display_name']} | {row['model_size']} | {row['task']} | "
                       f"{row['mse_mean']:.6f} ± {row['mse_std']:.6f} | "
                       f"{row['mae_mean']:.6f} ± {row['mae_std']:.6f} | "
                       f"{row['rmse_mean']:.6f} ± {row['rmse_std']:.6f} | "
                       f"{row['success_rate']:.1%} |\n")
            
            f.write("\n### Files Generated\n\n")
            f.write("- `detailed_statistics.csv`: Complete statistical summary\n")
            f.write("- `summary_statistics.csv`: Summary with mean and std for MSE and MAE\n")
            f.write("- `statistical_tests.csv`: Pairwise significance tests\n")
            f.write("- `performance_heatmaps.png`: Performance comparison visualizations\n")
            f.write("- `model_rankings.png`: Model ranking visualizations\n")
            f.write("- `prediction_comparisons.png`: Power prediction vs ground truth plots\n")
            f.write("- Individual experiment results in subdirectories\n")
            
        logger.info(f"Final report generated: {report_path}")

def main():
    """Main function to run the final comprehensive experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Final Comprehensive Power Prediction Experiment')
    parser.add_argument('--test', action='store_true', help='Run in test mode (2 models, 10 epochs)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs per experiment')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per configuration')
    args = parser.parse_args()
    
    if args.test:
        logger.info("Test Mode: Final Comprehensive Power Prediction Experiment")
        logger.info("=" * 60)
        # Test mode: 2 models, 10 epochs, 1 run
        experiment = FinalComprehensiveExperiment(base_epochs=10, num_runs=1, test_mode=True)
    else:
        logger.info("Full Mode: Final Comprehensive Power Prediction Experiment")
        logger.info("=" * 80)
        # Full mode: 7 models, specified epochs and runs
        experiment = FinalComprehensiveExperiment(base_epochs=args.epochs, num_runs=args.runs, test_mode=False)
    
    try:
        # Run all experiments
        experiment.run_all_experiments()
        
        # Analyze results
        experiment.analyze_results()
        
        logger.info("Experiment finished successfully!")
        logger.info(f"Results saved to: {experiment.results_dir}")
        
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        logger.info(f"Partial results saved to: {experiment.results_dir}")
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        logger.info(f"Partial results saved to: {experiment.results_dir}")
        raise

if __name__ == "__main__":
    main() 