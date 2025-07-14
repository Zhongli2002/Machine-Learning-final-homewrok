"""
MoE Model Training Script
Used to train the Mixture of Experts model for time series power prediction.
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
from datetime import datetime
warnings.filterwarnings('ignore')

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import PowerDataProcessor, create_data_loaders
from models.moe_model import create_moe_model
from config import get_config, get_device_config
from utils.training_utils import log_gradients, plot_gradient_flow


class MoETrainer:
    """MoE Trainer with task-aware training"""
    
    def __init__(self, config):
        self.config = config
        self.results_dir = config['results_dir']
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Device setup
        self.device, self.use_multi_gpu, self.gpu_ids = get_device_config(
            config.get('use_multi_gpu', True),
            config.get('gpu_ids', None)
        )
        
        torch.manual_seed(config['random_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config['random_seed'])
        
        # Data processor
        self.data_processor = PowerDataProcessor(scaler_type=config['scaler_type'])
        
        # Model components
        self.model: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.criterion = nn.MSELoss()
        self.train_losses, self.val_losses = [], []
        self.expert_weights_history = []
        self.gradient_history = {}
        
        # Task ID mapping
        self.task_id = 0 if config['prediction_length'] == 90 else 1  # 0: short, 1: long
    
    def prepare_data(self):
        """Prepare data (same as other trainers)"""
        print("Loading and preprocessing data...")
        train_df, test_df = self.data_processor.load_data(
            self.config['train_path'], self.config['test_path']
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
        
        X_train, y_train = self.data_processor.create_sequences(
            train_data, self.config['sequence_length'], self.config['prediction_length']
        )
        X_test, y_test = self.data_processor.create_sequences(
            test_data, self.config['sequence_length'], self.config['prediction_length']
        )
        
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        
        val_size = int(len(X_train) * self.config['val_split'])
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_train, y_train = X_train[:-val_size], y_train[:-val_size]
        
        self.train_loader, self.val_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size=self.config['batch_size']
        )
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        self.y_test = torch.FloatTensor(y_test).to(self.device)
    
    def build_model(self):
        """Build MoE model"""
        if self.data_processor.feature_columns is None:
            raise ValueError("Call prepare_data first")
        
        input_size = len(self.data_processor.feature_columns) + 1
        
        self.model = create_moe_model(
            input_size=input_size,
            output_length=self.config['prediction_length'],
            lstm_hidden_size=self.config['lstm_hidden_size'],
            lstm_num_layers=self.config['lstm_num_layers'],
            lstm_dropout=self.config['lstm_dropout'],
            transformer_d_model=self.config['transformer_d_model'],
            transformer_nhead=self.config['transformer_nhead'],
            transformer_num_layers=self.config['transformer_num_layers'],
            transformer_dropout=self.config['transformer_dropout'],
            gate_hidden_size=self.config['gate_hidden_size'],
        ).to(self.device)
        
        if self.use_multi_gpu and len(self.gpu_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"MoE parameters: {total_params:,}")
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate'], 
            weight_decay=self.config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', 
            patience=self.config.get('lr_patience', 10), 
            factor=0.5
        )
    
    def train_epoch(self, epoch):
        """Train one epoch with task-aware routing"""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in self.train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward with task ID
            outputs = self.model(batch_x, task_id=self.task_id)
            loss = self.criterion(outputs, batch_y)
            
            loss.backward()
            log_gradients(self.model, epoch, self.gradient_history)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x, task_id=self.task_id)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Train with early stopping"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 
                          os.path.join(self.results_dir, 'best_model.pth'))
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config['epochs']}, "
                      f"Train: {train_loss:.4f}, Val: {val_loss:.4f}")
            
            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        self.model.load_state_dict(torch.load(os.path.join(self.results_dir, 'best_model.pth')))
    
    def evaluate(self):
        """Evaluate the model on the test set."""
        assert self.model is not None
        print("Evaluating model...")
        self.model.eval()
        
        with torch.no_grad():
            predictions, gate_outputs = self.model(self.X_test, task_id=self.task_id, return_gate_weights=True)

        predictions_cpu = predictions.cpu().numpy()
        y_test_cpu = self.y_test.cpu().numpy()

        predictions_original = self.data_processor.inverse_transform_target(predictions_cpu)
        y_test_original = self.data_processor.inverse_transform_target(y_test_cpu)
        
        mse = mean_squared_error(y_test_original, predictions_original)
        mae = mean_absolute_error(y_test_original, predictions_original)
        rmse = np.sqrt(mse)
        
        print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        results = {'mse': mse, 'mae': mae, 'rmse': rmse}
        # Add MoE specific metrics if available
        if gate_outputs is not None:
            avg_gate_weights = gate_outputs.mean(dim=0).cpu().numpy()
            results['lstm_weight'] = float(avg_gate_weights[0])
            results['transformer_weight'] = float(avg_gate_weights[1])
            print(f"Average Gate Weights - LSTM: {results['lstm_weight']:.3f}, Transformer: {results['transformer_weight']:.3f}")
        
        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        np.savez_compressed(
            os.path.join(self.results_dir, 'predictions.npz'),
            predictions=predictions_original,
            ground_truth=y_test_original
        )
        print(f"Predictions and ground truth saved to {self.results_dir}/predictions.npz")
        
        return predictions_original, y_test_original
    
    def plot_results(self, predictions, actuals):
        """Plot results of predictions vs actuals"""
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 1, 1)
        plt.plot(actuals.flatten(), label='Actual Values', color='blue', alpha=0.7)
        plt.plot(predictions.flatten(), label='Predicted Values', color='red', linestyle='--')
        plt.title('Prediction vs Actual')
        plt.xlabel('Time Step')
        plt.ylabel('Power')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(self.results_dir, 'prediction_vs_actual.png')
        plt.savefig(plot_path)
        print(f"Saved plot to: {plot_path}")
        plt.close()
    
    def run(self):
        """Run complete training pipeline"""
        self.prepare_data()
        self.build_model()
        self.train()
        preds, acts = self.evaluate()
        self.plot_results(preds, acts)
        plot_gradient_flow(self.gradient_history, os.path.join(self.results_dir, 'gradient_flow.png'))


def run_experiment(config):
    """Run a single experiment."""
    trainer = MoETrainer(config)
    trainer.run()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='MoE Model Training')
    parser.add_argument('--task', type=str, default='short', choices=['short', 'long'], help='Task type')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--results_dir', type=str, default=None, help='Results directory')
    parser.add_argument('--model_size', type=str, default='standard', choices=['standard', 'large'], help='Model size')
    args = parser.parse_args()
    
    print(f"Running MoE model, Task: {args.task}, Model Size: {args.model_size}")

    config = get_config('moe', args.task, model_size=args.model_size)
    
    if args.epochs:
        config['epochs'] = args.epochs
    
    if args.results_dir:
        config['results_dir'] = args.results_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_size_str = f"_{args.model_size}" if args.model_size == 'large' else ""
        config['results_dir'] = f"results/moe_{args.task}{model_size_str}_{timestamp}"

    run_experiment(config)


if __name__ == "__main__":
    main() 