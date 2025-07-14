"""
MoE V2 Model Training Script - Enhanced Version
-----------------------------------------------
Training script for the ultimate MoE V2 model that combines:
- EnhancedLSTM v3.0: Superior long-term prediction expert
- EnhancedTransformer: Superior short-term prediction expert
- AdvancedGatingNetwork: Intelligent task-aware routing
- ExpertFusionLayer: Advanced expert combination

Enhanced training strategies:
1. Progressive training with curriculum learning
2. Adaptive loss weighting
3. Expert usage monitoring
4. Temperature annealing
5. Advanced regularization
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import argparse
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from datetime import datetime
warnings.filterwarnings('ignore')

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import PowerDataProcessor, create_data_loaders
from models.moe_v2_model import create_moe_v2_model
from config import get_config, get_device_config
from utils.training_utils import log_gradients, plot_gradient_flow


class AdvancedMoEV2Trainer:
    """Enhanced MoE V2 Trainer with advanced training strategies"""
    
    def __init__(self, config):
        self.config = config
        self.results_dir = config['results_dir']
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Device setup - Force single GPU for MoE V2 to avoid CUDA issues
        self.device, self.use_multi_gpu, self.gpu_ids = get_device_config(
            config.get('use_multi_gpu', False),  # Force single GPU
            config.get('gpu_ids', [0])  # Use only GPU 0
        )
        
        torch.manual_seed(config['random_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config['random_seed'])
        
        # Data processor
        self.data_processor = PowerDataProcessor(scaler_type=config['scaler_type'])
        
        # Model components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses, self.val_losses = [], []
        self.load_balance_losses = []
        self.expert_usage_history = []
        self.gradient_history = {}
        
        # Task ID mapping
        self.task_id = 0 if config['prediction_length'] == 90 else 1  # 0: short, 1: long
        
        # Advanced training parameters
        self.warmup_epochs = config.get('warmup_epochs', 20)
        self.load_balance_weight = config.get('load_balance_weight', 0.01)
        self.expert_diversity_weight = config.get('expert_diversity_weight', 0.005)
        
        print(f"Advanced MoE V2 Trainer initialized for {'short-term' if self.task_id == 0 else 'long-term'} prediction")
        print(f"Device: {self.device}, Multi-GPU: {self.use_multi_gpu}")
        print(f"Warmup epochs: {self.warmup_epochs}, Load balance weight: {self.load_balance_weight}")
    
    def prepare_data(self):
        """Prepare data for training"""
        print("Preparing data...")
        
        # Load and process data
        train_df, test_df = self.data_processor.load_data(
            self.config['train_path'],
            self.config['test_path']
        )
        
        train_df = self.data_processor.preprocess_data(train_df)
        test_df = self.data_processor.preprocess_data(test_df)
        train_df = self.data_processor.aggregate_daily_data(train_df)
        test_df = self.data_processor.aggregate_daily_data(test_df)
        train_df = self.data_processor.prepare_features(train_df)
        test_df = self.data_processor.prepare_features(test_df)
        
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
        self.data_processor.fit_scalers(train_df)
        train_data = self.data_processor.transform_data(train_df)
        test_data = self.data_processor.transform_data(test_df)
        
        # Create sequences
        X_train, y_train = self.data_processor.create_sequences(
            train_data,
            sequence_length=self.config['sequence_length'],
            prediction_length=self.config['prediction_length']
        )
        
        X_test, y_test = self.data_processor.create_sequences(
            test_data,
            sequence_length=self.config['sequence_length'],
            prediction_length=self.config['prediction_length']
        )
        
        print(f"Training sequences: {X_train.shape}, Test sequences: {X_test.shape}")
        
        # Create validation split
        val_size = int(len(X_train) * self.config['val_split'])
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_train, y_train = X_train[:-val_size], y_train[:-val_size]
        
        # Create data loaders
        self.train_loader, self.val_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, 
            batch_size=self.config['batch_size']
        )
        
        # Save test data
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        self.y_test = torch.FloatTensor(y_test).to(self.device)
        
        return train_df, test_df
    
    def _unwrap_model(self):
        """Utility to unwrap model from DataParallel if needed."""
        return getattr(self.model, 'module', self.model)
    
    def build_model(self):
        """Build enhanced MoE V2 model"""
        if self.data_processor.feature_columns is None:
            raise ValueError("Call prepare_data first")
        
        input_size = len(self.data_processor.feature_columns) + 1
        
        self.model = create_moe_v2_model(
            input_size=input_size,
            lstm_hidden_size=self.config['lstm_hidden_size'],
            lstm_num_layers=self.config['lstm_num_layers'],
            lstm_dropout=self.config['lstm_dropout'],
            transformer_d_model=self.config['transformer_d_model'],
            transformer_nhead=self.config['transformer_nhead'],
            transformer_num_layers=self.config['transformer_num_layers'],
            transformer_dropout=self.config['transformer_dropout'],
            gate_hidden_size=self.config['gate_hidden_size'],
            fusion_hidden_size=self.config.get('fusion_hidden_size', 64),
        ).to(self.device)
        
        # Completely disable multi-GPU for MoE V2 to avoid CUDA issues
        # if self.use_multi_gpu and len(self.gpu_ids) > 1:
        #     self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Enhanced MoE V2 parameters: {total_params:,}")
        
        # Advanced optimizer with different learning rates for different components
        gating_params = []
        fusion_params = []
        expert_params = []
        
        for name, param in self.model.named_parameters():
            if 'gating' in name:
                gating_params.append(param)
            elif 'fusion' in name:
                fusion_params.append(param)
            else:
                expert_params.append(param)
        
        # Use different learning rates for different components
        self.optimizer = optim.AdamW([
            {'params': expert_params, 'lr': self.config['learning_rate'] * 0.8, 'weight_decay': self.config['weight_decay']},
            {'params': gating_params, 'lr': self.config['learning_rate'] * 1.2, 'weight_decay': self.config['weight_decay'] * 0.5},
            {'params': fusion_params, 'lr': self.config['learning_rate'], 'weight_decay': self.config['weight_decay'] * 0.1}
        ], eps=1e-8, betas=(0.9, 0.999))
        
        # Advanced learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config['epochs'], 
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        print("Enhanced MoE V2 model built successfully!")
        print(f"Expert parameters: {len(expert_params)}")
        print(f"Gating parameters: {len(gating_params)}")
        print(f"Fusion parameters: {len(fusion_params)}")
    
    def compute_advanced_loss(self, outputs, targets, gate_weights, load_balance_loss, epoch):
        """Compute advanced loss with multiple components"""
        # Primary prediction loss
        prediction_loss = self.criterion(outputs, targets)
        
        # Load balance loss (encourage balanced expert usage)
        load_balance_component = self.load_balance_weight * load_balance_loss
        
        # Expert diversity loss (encourage different experts for different patterns)
        diversity_loss = torch.tensor(0.0, device=outputs.device)
        if epoch > self.warmup_epochs:
            # Encourage diversity in gating decisions
            gate_entropy = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=1)
            diversity_loss = self.expert_diversity_weight * torch.mean(gate_entropy)
        
        # Progressive weighting of auxiliary losses
        aux_weight = min(1.0, epoch / self.warmup_epochs)
        total_loss = prediction_loss + aux_weight * (load_balance_component + diversity_loss)
        
        # Ensure all components are scalars
        total_loss = total_loss.mean() if total_loss.dim() > 0 else total_loss
        prediction_loss = prediction_loss.mean() if prediction_loss.dim() > 0 else prediction_loss
        load_balance_component = load_balance_component.mean() if load_balance_component.dim() > 0 else load_balance_component
        diversity_loss = diversity_loss.mean() if diversity_loss.dim() > 0 else diversity_loss
        
        return total_loss, prediction_loss, load_balance_component, diversity_loss
    
    def train_epoch(self, epoch):
        """Train for one epoch with advanced strategies"""
        self.model.train()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_balance_loss = 0.0
        total_diversity_loss = 0.0
        
        expert_usage_batch = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with gate weights
            outputs, gate_weights, load_balance_loss = self.model(
                data, 
                task_id=self.task_id, 
                return_gate_weights=True,
                epoch=epoch,
                max_epochs=self.config['epochs']
            )
            
            # Compute advanced loss
            loss, pred_loss, balance_loss, diversity_loss = self.compute_advanced_loss(
                outputs, target, gate_weights, load_balance_loss, epoch
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_balance_loss += balance_loss.item()
            total_diversity_loss += diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss
            
            # Track expert usage
            dominant_expert = gate_weights.argmax(dim=1)
            lstm_usage = (dominant_expert == 0).float().mean().item()
            transformer_usage = (dominant_expert == 1).float().mean().item()
            expert_usage_batch.append({'lstm': lstm_usage, 'transformer': transformer_usage})
        
        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        avg_pred_loss = total_pred_loss / len(self.train_loader)
        avg_balance_loss = total_balance_loss / len(self.train_loader)
        avg_diversity_loss = total_diversity_loss / len(self.train_loader)
        
        # Average expert usage
        avg_expert_usage = {
            'lstm': np.mean([b['lstm'] for b in expert_usage_batch]),
            'transformer': np.mean([b['transformer'] for b in expert_usage_batch])
        }
        
        return avg_loss, avg_pred_loss, avg_balance_loss, avg_diversity_loss, avg_expert_usage
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = self.model(
                    data, 
                    task_id=self.task_id,
                    epoch=epoch,
                    max_epochs=self.config['epochs']
                )
                
                loss = self.criterion(outputs, target)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Enhanced training loop"""
        print("Starting enhanced MoE V2 training...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training
            train_loss, pred_loss, balance_loss, diversity_loss, expert_usage = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate_epoch(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Track history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.load_balance_losses.append(balance_loss)
            self.expert_usage_history.append(expert_usage)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), os.path.join(self.results_dir, 'best_model.pth'))
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0 or epoch == self.config['epochs'] - 1:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{self.config['epochs']}:")
                print(f"  Train Loss: {train_loss:.6f} (Pred: {pred_loss:.6f}, Balance: {balance_loss:.6f}, Diversity: {diversity_loss:.6f})")
                print(f"  Val Loss: {val_loss:.6f}, Best: {best_val_loss:.6f}")
                print(f"  Expert Usage - LSTM: {expert_usage['lstm']:.3f}, Transformer: {expert_usage['transformer']:.3f}")
                print(f"  Learning Rate: {current_lr:.6f}")
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load(os.path.join(self.results_dir, 'best_model.pth')))
        print("Enhanced MoE V2 training completed!")
    
    def evaluate(self):
        """Enhanced evaluation with expert analysis"""
        print("Evaluating enhanced MoE V2...")
        
        self.model.eval()
        predictions = []
        gate_weights_all = []
        
        with torch.no_grad():
            for i in range(0, len(self.X_test), self.config['batch_size']):
                batch_X = self.X_test[i:i+self.config['batch_size']]
                
                # Get predictions with gate weights
                batch_pred, batch_gates, _ = self.model(
                    batch_X, 
                    task_id=self.task_id,
                    return_gate_weights=True
                )
                
                predictions.append(batch_pred.cpu().numpy())
                gate_weights_all.append(batch_gates.cpu().numpy())
        
        # Combine predictions
        predictions = np.concatenate(predictions, axis=0)
        gate_weights_all = np.concatenate(gate_weights_all, axis=0)
        
        # Inverse transform
        y_test_original = self.data_processor.inverse_transform_target(self.y_test.cpu().numpy())
        predictions_original = self.data_processor.inverse_transform_target(predictions)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_original, predictions_original)
        mae = mean_absolute_error(y_test_original, predictions_original)
        rmse = np.sqrt(mse)
        
        # Expert usage analysis
        lstm_usage = (gate_weights_all.argmax(axis=1) == 0).mean()
        transformer_usage = (gate_weights_all.argmax(axis=1) == 1).mean()
        
        # Gate weights statistics
        avg_lstm_weight = gate_weights_all[:, 0].mean()
        avg_transformer_weight = gate_weights_all[:, 1].mean()
        
        print(f"Enhanced MoE V2 Test Results:")
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"Expert Usage - LSTM: {lstm_usage:.3f}, Transformer: {transformer_usage:.3f}")
        print(f"Average Gate Weights - LSTM: {avg_lstm_weight:.3f}, Transformer: {avg_transformer_weight:.3f}")
        
        # Save results
        results = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'lstm_weight': float(avg_lstm_weight),
            'transformer_weight': float(avg_transformer_weight),
            'lstm_usage': float(lstm_usage),
            'transformer_usage': float(transformer_usage),
            'config': self.config
        }
        
        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'load_balance_losses': self.load_balance_losses,
            'expert_usage_history': self.expert_usage_history
        }
        
        with open(os.path.join(self.results_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save predictions and ground truth
        np.savez_compressed(
            os.path.join(self.results_dir, 'predictions.npz'),
            predictions=predictions_original,
            ground_truth=y_test_original
        )
        print(f"Predictions and ground truth saved to {self.results_dir}/predictions.npz")
        
        # Create enhanced visualizations
        self.create_enhanced_visualizations(predictions_original, y_test_original, gate_weights_all)
        
        return results
    
    def create_enhanced_visualizations(self, predictions, targets, gate_weights):
        """Create enhanced visualizations"""
        # Training curves
        plt.figure(figsize=(15, 12))
        
        # Loss curves
        plt.subplot(2, 3, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Load balance loss
        plt.subplot(2, 3, 2)
        plt.plot(self.load_balance_losses, label='Load Balance Loss')
        plt.title('Load Balance Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Expert usage over time
        plt.subplot(2, 3, 3)
        lstm_usage = [h['lstm'] for h in self.expert_usage_history]
        transformer_usage = [h['transformer'] for h in self.expert_usage_history]
        plt.plot(lstm_usage, label='LSTM Usage')
        plt.plot(transformer_usage, label='Transformer Usage')
        plt.title('Expert Usage Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Usage Ratio')
        plt.legend()
        plt.grid(True)
        
        # Gate weights distribution
        plt.subplot(2, 3, 4)
        plt.hist(gate_weights[:, 0], bins=50, alpha=0.7, label='LSTM Weights')
        plt.hist(gate_weights[:, 1], bins=50, alpha=0.7, label='Transformer Weights')
        plt.title('Gate Weights Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        
        # Prediction vs Target
        plt.subplot(2, 3, 5)
        sample_indices = np.random.choice(len(predictions), min(500, len(predictions)), replace=False)
        plt.scatter(targets[sample_indices], predictions[sample_indices], alpha=0.6)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        plt.title('Predictions vs Targets')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True)
        
        # Error distribution
        plt.subplot(2, 3, 6)
        errors = predictions - targets
        plt.hist(errors.flatten(), bins=50, alpha=0.7)
        plt.title('Prediction Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'enhanced_training_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Enhanced visualizations saved to {self.results_dir}")


def run_experiment(config):
    """Run enhanced MoE V2 experiment"""
    trainer = AdvancedMoEV2Trainer(config)
    
    # Prepare data
    trainer.prepare_data()
    
    # Build model
    trainer.build_model()
    
    # Train
    trainer.train()
    
    # Evaluate
    results = trainer.evaluate()
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Enhanced MoE V2 Experiment')
    parser.add_argument('--task', type=str, default='short', choices=['short', 'long'], help='Task type for the experiment')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--model_size', type=str, default='standard', choices=['standard', 'large'], help='Model size to run (standard or large)')
    parser.add_argument('--results_dir', type=str, default=None, help='Directory to save results')
    args = parser.parse_args()
    
    # Get configuration based on task and model size
    config = get_config(f'moe_v2', args.task, model_size=args.model_size)
    
    # Override config with command line arguments if provided
    if args.epochs:
        config['epochs'] = args.epochs

    if args.results_dir:
        config['results_dir'] = args.results_dir
    else:
        # Create a default results directory if not provided
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_size_str = f"_{args.model_size}" if args.model_size == 'large' else ""
        config['results_dir'] = os.path.join('results', f"moe_v2_{args.task}{model_size_str}_{timestamp}")

    print(f"Running Enhanced MoE V2, Task: {args.task}, Model Size: {args.model_size}")
    
    # Run the experiment
    run_experiment(config)


if __name__ == "__main__":
    main() 