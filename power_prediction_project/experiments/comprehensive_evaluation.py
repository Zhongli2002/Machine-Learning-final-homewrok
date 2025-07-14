"""
Comprehensive Model Evaluation - Ablation Study
----------------------------------------------
This script runs all models in our project for a complete performance comparison:

Original Models:
- LSTM (basic)
- Transformer (encoder)
- Innovative (CNN-LSTM-Attention hybrid)
- Informer

Enhanced Models:
- Enhanced LSTM v3.0
- Enhanced Transformer

Mixture of Experts:
- MoE V1 (original)
- MoE V2 (with enhanced experts)

This provides the complete ablation study showing the progression from
basic models to our ultimate MoE V2 architecture.
"""

import os
import sys
import json
import subprocess
import time
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_model_experiment(model_name, task_type, epochs=None, results_base_dir=None):
    """
    Run a single model experiment.
    
    Args:
        model_name: Name of the model to run
        task_type: 'short_term' or 'long_term'
        epochs: Number of epochs (optional override)
        results_base_dir: Base directory for results (optional)
    
    Returns:
        dict: Results including MSE, MAE, RMSE and other metrics
    """
    print(f"\n{'='*60}")
    print(f"Running {model_name.upper()} - {task_type.replace('_', '-')} prediction")
    print(f"{'='*60}")
    
    # Determine the correct script to run
    script_map = {
        'lstm': 'train_lstm.py',
        'transformer': 'train_transformer.py', 
        'innovative': 'train_innovative.py',
        'informer': 'train_informer.py',
        'moe': 'train_moe.py',
        'enhanced_lstm': 'train_enhanced_lstm.py',
        'enhanced_transformer': 'train_enhanced_transformer.py',
        'moe_v2': 'train_moe_v2.py'
    }
    
    if model_name not in script_map:
        print(f"Unknown model: {model_name}")
        return None
    
    script_name = script_map[model_name]
    
    # Build command
    cmd_parts = ['python', script_name]
    
    # Handle different argument formats for different scripts
    if model_name == 'moe_v2':
        # MoE V2 uses --task with 'short'/'long' values
        task_arg = 'short' if task_type == 'short_term' else 'long'
        cmd_parts.append(f'--task={task_arg}')
    else:
        # Other scripts use --task_type with 'short_term'/'long_term' values
        cmd_parts.append(f'--task_type={task_type}')
    
    if epochs is not None:
        cmd_parts.append(f'--epochs={epochs}')
    
    if results_base_dir is not None:
        result_dir = os.path.join(results_base_dir, f'{model_name}_{task_type}')
        cmd_parts.append(f'--results_dir={result_dir}')
    
    command = ' '.join(cmd_parts)
    
    try:
        start_time = time.time()
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True, cwd='.')
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"✓ Completed in {duration:.1f} seconds")
        
        # Try to load results
        if results_base_dir:
            results_file = os.path.join(results_base_dir, f'{model_name}_{task_type}', 'results.json')
        else:
            # Use default results directory structure
            from config import get_config
            config = get_config(model_name, task_type.split('_')[0])
            results_file = os.path.join(config['results_dir'], 'results.json')
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            results['duration'] = duration
            results['success'] = True
            return results
        else:
            print(f"⚠ Results file not found: {results_file}")
            return {'success': False, 'duration': duration, 'error': 'Results file not found'}
            
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"✗ Failed after {duration:.1f} seconds")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout[-500:]}")  # Last 500 chars
        return {'success': False, 'duration': duration, 'error': str(e)}


def create_results_summary(all_results, output_file):
    """Create a comprehensive results summary."""
    
    # Prepare data for DataFrame
    summary_data = []
    
    for model_name, tasks in all_results.items():
        for task_type, results in tasks.items():
            if results and results.get('success', False):
                row = {
                    'Model': model_name,
                    'Task': task_type.replace('_', '-'),
                    'MSE': results.get('mse', np.nan),
                    'MAE': results.get('mae', np.nan), 
                    'RMSE': results.get('rmse', np.nan),
                    'Duration (s)': results.get('duration', np.nan),
                    'Success': '✓'
                }
                
                # Add gate weights for MoE models
                if 'avg_gate_weights' in results:
                    weights = results['avg_gate_weights']
                    if len(weights) >= 2:
                        row['LSTM_Weight'] = weights[0]
                        row['Transformer_Weight'] = weights[1]
                        row['Gating_Correct'] = '✓' if results.get('gating_behavior_correct', False) else '✗'
                
                summary_data.append(row)
            else:
                # Failed experiment
                row = {
                    'Model': model_name,
                    'Task': task_type.replace('_', '-'),
                    'MSE': np.nan,
                    'MAE': np.nan,
                    'RMSE': np.nan,
                    'Duration (s)': results.get('duration', np.nan) if results else np.nan,
                    'Success': '✗'
                }
                summary_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    # Sort by task and then by MSE (best first)
    df_sorted = df.sort_values(['Task', 'MSE'], na_position='last')
    
    # Save to CSV
    csv_file = output_file.replace('.txt', '.csv')
    df_sorted.to_csv(csv_file, index=False)
    
    # Create formatted text summary
    with open(output_file, 'w') as f:
        f.write("COMPREHENSIVE MODEL EVALUATION - ABLATION STUDY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Group by task
        for task in ['short-term', 'long-term']:
            task_data = df_sorted[df_sorted['Task'] == task]
            if len(task_data) == 0:
                continue
                
            f.write(f"\n{task.upper()} PREDICTION RESULTS\n")
            f.write("-" * 40 + "\n")
            
            # Find best performing model
            successful_models = task_data[task_data['Success'] == '✓']
            if len(successful_models) > 0:
                best_model = successful_models.loc[successful_models['MSE'].idxmin()]
                f.write(f"🏆 BEST MODEL: {best_model['Model']} (MSE: {best_model['MSE']:.2f})\n\n")
            
            # Write detailed results
            for _, row in task_data.iterrows():
                f.write(f"{row['Model'].upper():<20} ")
                if row['Success'] == '✓':
                    f.write(f"MSE: {row['MSE']:>10.2f}  MAE: {row['MAE']:>8.2f}  RMSE: {row['RMSE']:>8.2f}")
                    if not pd.isna(row.get('LSTM_Weight')):
                        f.write(f"  Gates: [{row['LSTM_Weight']:.3f}, {row['Transformer_Weight']:.3f}] {row.get('Gating_Correct', '')}")
                else:
                    f.write("FAILED")
                f.write(f"  ({row['Duration (s)']:.1f}s)\n")
        
        # Model category analysis
        f.write(f"\n\nMODEL CATEGORY ANALYSIS\n")
        f.write("=" * 30 + "\n")
        
        categories = {
            'Original Models': ['lstm', 'transformer', 'innovative', 'informer'],
            'Enhanced Models': ['enhanced_lstm', 'enhanced_transformer'], 
            'Mixture of Experts': ['moe', 'moe_v2']
        }
        
        for category, models in categories.items():
            f.write(f"\n{category}:\n")
            category_data = df_sorted[df_sorted['Model'].isin(models)]
            successful_category = category_data[category_data['Success'] == '✓']
            
            if len(successful_category) > 0:
                best_short = successful_category[successful_category['Task'] == 'short-term']
                best_long = successful_category[successful_category['Task'] == 'long-term']
                
                if len(best_short) > 0:
                    best_short_model = best_short.loc[best_short['MSE'].idxmin()]
                    f.write(f"  Best Short-term: {best_short_model['Model']} (MSE: {best_short_model['MSE']:.2f})\n")
                
                if len(best_long) > 0:
                    best_long_model = best_long.loc[best_long['MSE'].idxmin()]
                    f.write(f"  Best Long-term:  {best_long_model['Model']} (MSE: {best_long_model['MSE']:.2f})\n")
            else:
                f.write("  No successful experiments\n")
        
        f.write(f"\nDetailed results saved to: {csv_file}\n")
    
    print(f"\nResults summary saved to: {output_file}")
    print(f"Detailed CSV saved to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--models', nargs='+', 
                       choices=['lstm', 'transformer', 'innovative', 'informer', 'moe',
                               'enhanced_lstm', 'enhanced_transformer', 'moe_v2', 'all'],
                       default=['all'],
                       help='Models to evaluate')
    parser.add_argument('--tasks', nargs='+',
                       choices=['short_term', 'long_term', 'all'],
                       default=['all'], 
                       help='Tasks to run')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override epochs for all experiments')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Base directory for results')
    parser.add_argument('--output', type=str, default='comprehensive_evaluation_results.txt',
                       help='Output file for summary')
    
    args = parser.parse_args()
    
    # Determine models to run
    if 'all' in args.models:
        models_to_run = ['lstm', 'transformer', 'innovative', 'informer', 'moe',
                        'enhanced_lstm', 'enhanced_transformer', 'moe_v2']
    else:
        models_to_run = args.models
    
    # Determine tasks to run
    if 'all' in args.tasks:
        tasks_to_run = ['short_term', 'long_term']
    else:
        tasks_to_run = args.tasks
    
    print("🚀 COMPREHENSIVE MODEL EVALUATION STARTING")
    print(f"Models: {', '.join(models_to_run)}")
    print(f"Tasks: {', '.join(tasks_to_run)}")
    if args.epochs:
        print(f"Epochs: {args.epochs}")
    print()
    
    # Run all experiments
    all_results = {}
    total_experiments = len(models_to_run) * len(tasks_to_run)
    current_experiment = 0
    
    start_time = time.time()
    
    for model_name in models_to_run:
        all_results[model_name] = {}
        
        for task_type in tasks_to_run:
            current_experiment += 1
            print(f"\n[{current_experiment}/{total_experiments}] Starting {model_name} - {task_type}")
            
            results = run_model_experiment(
                model_name=model_name,
                task_type=task_type,
                epochs=args.epochs,
                results_base_dir=args.results_dir
            )
            
            all_results[model_name][task_type] = results
    
    total_time = time.time() - start_time
    
    # Create comprehensive summary
    create_results_summary(all_results, args.output)
    
    print(f"\nEVALUATION COMPLETE!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main() 