#!/usr/bin/env python3
"""
Comprehensive Prediction Comparison Visualization
===============================================

This script creates a comprehensive visualization showing all model predictions
compared to ground truth in a single figure.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensivePredictionVisualizer:
    def __init__(self, results_dir='results_final/final_experiment_20250713_015024'):
        self.results_dir = results_dir
        self.models = {
            'lstm': 'LSTM',
            'transformer': 'Transformer', 
            'informer': 'Informer',
            'moe_v2': 'MoE V2',
            'enhanced_lstm': 'Enhanced LSTM',
            'enhanced_transformer': 'Enhanced Transformer',
            'innovative': 'Innovative Model'
        }
        self.tasks = ['short', 'long']
        self.sizes = ['standard', 'large']
        
    def load_best_predictions(self):
        """Load the best performing model predictions for each configuration."""
        best_predictions = {}
        
        for model_key, model_name in self.models.items():
            for task in self.tasks:
                for size in self.sizes:
                    # Find all runs for this configuration
                    pattern = f"{model_key}_{task}_{size}_run"
                    run_dirs = [d for d in os.listdir(self.results_dir) 
                               if d.startswith(pattern) and os.path.isdir(os.path.join(self.results_dir, d))]
                    
                    if not run_dirs:
                        continue
                    
                    # Find the best run (lowest MSE)
                    best_mse = float('inf')
                    best_run_dir = None
                    
                    for run_dir in run_dirs:
                        results_file = os.path.join(self.results_dir, run_dir, 'results.json')
                        if os.path.exists(results_file):
                            with open(results_file, 'r') as f:
                                results = json.load(f)
                                mse = results.get('mse', float('inf'))
                                if mse < best_mse:
                                    best_mse = mse
                                    best_run_dir = run_dir
                    
                    # Load predictions from best run
                    if best_run_dir:
                        pred_file = os.path.join(self.results_dir, best_run_dir, 'predictions.npz')
                        if os.path.exists(pred_file):
                            data = np.load(pred_file)
                            predictions = data['predictions']
                            ground_truth = data['ground_truth']
                            
                            config_key = f"{model_name}_{task}_{size}"
                            best_predictions[config_key] = {
                                'predictions': predictions,
                                'ground_truth': ground_truth,
                                'mse': best_mse,
                                'model_name': model_name,
                                'task': task,
                                'size': size
                            }
        
        return best_predictions
    
    def create_comprehensive_comparison(self, best_predictions, save_path='comprehensive_prediction_comparison.png'):
        """Create a comprehensive comparison plot with all models."""
        # Create subplots: 2 rows (short/long) x 2 columns (standard/large)
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Comprehensive Model Prediction Comparison', fontsize=16, fontweight='bold')
        
        # Define colors for each model
        colors = {
            'LSTM': '#1f77b4',
            'Transformer': '#ff7f0e', 
            'Informer': '#2ca02c',
            'MoE V2': '#d62728',
            'Enhanced LSTM': '#9467bd',
            'Enhanced Transformer': '#8c564b',
            'Innovative Model': '#e377c2'
        }
        
        # Plot configurations
        configs = [
            ('short', 'standard', 0, 0),
            ('short', 'large', 0, 1),
            ('long', 'standard', 1, 0),
            ('long', 'large', 1, 1)
        ]
        
        for task, size, row, col in configs:
            ax = axes[row, col]
            
            # Get all models for this configuration
            config_models = [(k, v) for k, v in best_predictions.items() 
                           if v['task'] == task and v['size'] == size]
            
            if not config_models:
                ax.set_title(f'No data for {task.title()} {size.title()}')
                continue
            
            # Plot ground truth first
            ground_truth = None
            for _, data in config_models:
                if ground_truth is None:
                    ground_truth = data['ground_truth']
                    break
            
            if ground_truth is not None:
                # Use only first 100 points for better visualization
                n_points = min(100, len(ground_truth))
                x = np.arange(n_points)
                
                # Plot ground truth
                ax.plot(x, ground_truth[:n_points], 'k-', linewidth=2, 
                       label='Ground Truth', alpha=0.8)
                
                # Plot each model's predictions
                for config_key, data in config_models:
                    model_name = data['model_name']
                    predictions = data['predictions']
                    mse = data['mse']
                    
                    color = colors.get(model_name, '#000000')
                    ax.plot(x, predictions[:n_points], '--', 
                           color=color, linewidth=1.5, alpha=0.7,
                           label=f'{model_name} (MSE: {mse:.0f})')
            
            # Customize subplot
            ax.set_title(f'{task.title()}-term Prediction ({size.title()} Models)', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Power Consumption (kW)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive comparison plot saved to: {save_path}")
        return save_path
    
    def create_best_models_comparison(self, best_predictions, save_path='best_models_comparison.png'):
        """Create a comparison of the best performing models only."""
        # Find best models for each task
        best_models = {}
        
        for task in self.tasks:
            task_models = [(k, v) for k, v in best_predictions.items() if v['task'] == task]
            if task_models:
                best_key, best_data = min(task_models, key=lambda x: x[1]['mse'])
                best_models[task] = (best_key, best_data)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Best Model Predictions Comparison', fontsize=16, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (task, (best_key, best_data)) in enumerate(best_models.items()):
            ax = axes[i]
            
            ground_truth = best_data['ground_truth']
            predictions = best_data['predictions']
            model_name = best_data['model_name']
            size = best_data['size']
            mse = best_data['mse']
            
            # Use first 150 points for better visualization
            n_points = min(150, len(ground_truth))
            x = np.arange(n_points)
            
            # Plot ground truth and predictions
            ax.plot(x, ground_truth[:n_points], 'k-', linewidth=2, 
                   label='Ground Truth', alpha=0.8)
            ax.plot(x, predictions[:n_points], '--', 
                   color=colors[i], linewidth=2, alpha=0.8,
                   label=f'{model_name} ({size.title()})')
            
            # Fill area between curves to show error
            ax.fill_between(x, ground_truth[:n_points], predictions[:n_points], 
                          alpha=0.2, color=colors[i])
            
            # Customize subplot
            ax.set_title(f'Best {task.title()}-term Prediction\n'
                        f'{model_name} ({size.title()}) - MSE: {mse:.0f}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Power Consumption (kW)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Best models comparison plot saved to: {save_path}")
        return save_path
    
    def create_model_performance_summary(self, best_predictions, save_path='model_performance_summary.png'):
        """Create a summary visualization of model performance."""
        # Prepare data for visualization
        data = []
        for config_key, pred_data in best_predictions.items():
            data.append({
                'Model': pred_data['model_name'],
                'Task': pred_data['task'].title(),
                'Size': pred_data['size'].title(),
                'MSE': pred_data['mse'],
                'Config': f"{pred_data['model_name']} ({pred_data['size'].title()})"
            })
        
        df = pd.DataFrame(data)
        
        # Create performance summary plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')
        
        # 1. MSE by Model and Task
        ax1 = axes[0, 0]
        pivot_data = df.pivot_table(values='MSE', index='Model', columns='Task', aggfunc='mean')
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='RdYlBu_r', ax=ax1)
        ax1.set_title('Average MSE by Model and Task')
        
        # 2. MSE by Model and Size
        ax2 = axes[0, 1]
        pivot_data2 = df.pivot_table(values='MSE', index='Model', columns='Size', aggfunc='mean')
        sns.heatmap(pivot_data2, annot=True, fmt='.0f', cmap='RdYlBu_r', ax=ax2)
        ax2.set_title('Average MSE by Model and Size')
        
        # 3. Bar plot of best configurations
        ax3 = axes[1, 0]
        best_configs = df.loc[df.groupby('Task')['MSE'].idxmin()]
        bars = ax3.bar(best_configs['Task'], best_configs['MSE'], 
                      color=['#1f77b4', '#ff7f0e'])
        ax3.set_title('Best MSE by Task')
        ax3.set_ylabel('MSE')
        
        # Add value labels on bars
        for bar, value in zip(bars, best_configs['MSE']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                    f'{value:.0f}', ha='center', va='bottom')
        
        # 4. Model ranking
        ax4 = axes[1, 1]
        model_avg_mse = df.groupby('Model')['MSE'].mean().sort_values()
        bars = ax4.barh(range(len(model_avg_mse)), model_avg_mse.values)
        ax4.set_yticks(range(len(model_avg_mse)))
        ax4.set_yticklabels(model_avg_mse.index)
        ax4.set_title('Average MSE by Model (Lower is Better)')
        ax4.set_xlabel('Average MSE')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, model_avg_mse.values)):
            ax4.text(bar.get_width() + 2000, bar.get_y() + bar.get_height()/2,
                    f'{value:.0f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance summary plot saved to: {save_path}")
        return save_path
    
    def generate_all_visualizations(self):
        """Generate all visualization plots."""
        print("Loading best predictions...")
        best_predictions = self.load_best_predictions()
        
        if not best_predictions:
            print("No prediction data found!")
            return
        
        print(f"Found {len(best_predictions)} model configurations")
        
        # Create output directory
        output_dir = 'prediction_visualizations'
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all plots
        plots = []
        
        print("Creating comprehensive comparison...")
        plot1 = self.create_comprehensive_comparison(
            best_predictions, 
            os.path.join(output_dir, 'comprehensive_prediction_comparison.png')
        )
        plots.append(plot1)
        
        print("Creating best models comparison...")
        plot2 = self.create_best_models_comparison(
            best_predictions,
            os.path.join(output_dir, 'best_models_comparison.png')
        )
        plots.append(plot2)
        
        print("Creating performance summary...")
        plot3 = self.create_model_performance_summary(
            best_predictions,
            os.path.join(output_dir, 'model_performance_summary.png')
        )
        plots.append(plot3)
        
        # Generate summary report
        self.generate_summary_report(best_predictions, output_dir)
        
        print(f"\nAll visualizations completed!")
        print(f"Output directory: {output_dir}")
        print("Generated files:")
        for plot in plots:
            print(f"  - {plot}")
        
        return plots
    
    def generate_summary_report(self, best_predictions, output_dir):
        """Generate a summary report of the best predictions."""
        report_path = os.path.join(output_dir, 'prediction_summary_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Comprehensive Prediction Comparison Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Best Model Performance\n\n")
            
            # Find best models for each task
            for task in self.tasks:
                task_models = [(k, v) for k, v in best_predictions.items() if v['task'] == task]
                if task_models:
                    best_key, best_data = min(task_models, key=lambda x: x[1]['mse'])
                    f.write(f"### {task.title()}-term Prediction\n")
                    f.write(f"- **Best Model:** {best_data['model_name']} ({best_data['size'].title()})\n")
                    f.write(f"- **MSE:** {best_data['mse']:.2f}\n")
                    f.write(f"- **Configuration:** {best_key}\n\n")
            
            f.write("## All Model Configurations\n\n")
            f.write("| Model | Task | Size | MSE |\n")
            f.write("|-------|------|------|-----|\n")
            
            # Sort by MSE
            sorted_configs = sorted(best_predictions.items(), key=lambda x: x[1]['mse'])
            
            for config_key, data in sorted_configs:
                f.write(f"| {data['model_name']} | {data['task'].title()} | {data['size'].title()} | {data['mse']:.2f} |\n")
        
        print(f"Summary report saved to: {report_path}")


def main():
    """Main function to generate all prediction visualizations."""
    print("Comprehensive Prediction Comparison Generator")
    print("=" * 50)
    
    # Check if results directory exists
    results_dir = 'results_final/final_experiment_20250713_015024'
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    # Create visualizer and generate plots
    visualizer = ComprehensivePredictionVisualizer(results_dir)
    plots = visualizer.generate_all_visualizations()
    
    print("\nVisualization generation completed successfully!")


if __name__ == "__main__":
    main() 