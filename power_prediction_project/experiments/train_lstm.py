"""
LSTM Model Training Script
Used to train the LSTM model for time series power prediction.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import argparse
from datetime import datetime
warnings.filterwarnings('ignore')

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import PowerDataProcessor, create_data_loaders
from models.lstm_model import create_lstm_model
from config import get_config, get_device_config
from utils.training_utils import log_gradients, plot_gradient_flow


class LSTMTrainer:
    """LSTM Trainer"""
    
    def __init__(self, config):
        """
        Initialize the trainer.
        
        Args:
            config (dict): Configuration parameters.
        """
        self.config = config
        self.results_dir = config['results_dir']
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set up device and multi-GPU support
        self.device, self.use_multi_gpu, self.gpu_ids = get_device_config()
        
        # Set random seed
        torch.manual_seed(config['random_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['random_seed'])
        
        # Initialize data processor
        self.data_processor = PowerDataProcessor(
            scaler_type=config['scaler_type']
        )
        
        # Initialize model
        self.model: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.gradient_history = {}
        
    def prepare_data(self):
        """Prepare the training data."""
        print("Loading and preprocessing data...")
        
        # Load data
        train_df, test_df = self.data_processor.load_data(
            self.config['train_path'], 
            self.config['test_path']
        )
        
        # Preprocess data
        train_df = self.data_processor.preprocess_data(train_df)
        test_df = self.data_processor.preprocess_data(test_df)
        
        # Aggregate data daily
        train_df = self.data_processor.aggregate_daily_data(train_df)
        test_df = self.data_processor.aggregate_daily_data(test_df)
        
        # Prepare features
        train_df = self.data_processor.prepare_features(train_df)
        test_df = self.data_processor.prepare_features(test_df)
        
        print(f"Train data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        # Fit scalers
        self.data_processor.fit_scalers(train_df)
        
        # Scale data
        train_data = self.data_processor.transform_data(train_df)
        test_data = self.data_processor.transform_data(test_df)
        
        # Create sequence data
        sequence_length = self.config['sequence_length']
        prediction_length = self.config['prediction_length']
        
        X_train, y_train = self.data_processor.create_sequences(
            train_data, sequence_length, prediction_length
        )
        X_test, y_test = self.data_processor.create_sequences(
            test_data, sequence_length, prediction_length
        )
        
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        # Split into training and validation sets
        val_split = self.config['val_split']
        val_size = int(len(X_train) * val_split)
        
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
    
    def build_model(self):
        """Build the LSTM model."""
        print("Building model...")
        
        # Get input feature dimension
        if self.data_processor.feature_columns is None:
            raise ValueError("Feature columns are not set. Call prepare_data first.")
        input_size = len(self.data_processor.feature_columns) + 1  # +1 for target
        
        # Create model
        self.model = create_lstm_model(
            input_size=input_size,
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            output_size=self.config['prediction_length'],
            dropout=self.config['dropout']
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Multi-GPU support
        if self.use_multi_gpu and len(self.gpu_ids) > 1:
            print(f"Using DataParallel to wrap the model, GPUs: {self.gpu_ids}")
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
        
        # Calculate number of model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Create optimizer and learning rate scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.config.get('lr_patience', 10),
            min_lr=1e-6
        )
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        assert self.model is not None
        assert self.optimizer is not None
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient logging
            log_gradients(self.model, epoch, self.gradient_history)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model."""
        assert self.model is not None
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Train the model."""
        print("Starting training...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Record losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduler
            assert self.optimizer is not None
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                assert self.model is not None
                torch.save(self.model.state_dict(), 
                          os.path.join(self.results_dir, 'best_model.pth'))
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config['epochs']}, "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Load the best model
        assert self.model is not None
        self.model.load_state_dict(torch.load(os.path.join(self.results_dir, 'best_model.pth')))
    
    def evaluate(self):
        """Evaluate the model on the test set."""
        assert self.model is not None
        print("Evaluating model...")
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(self.X_test)

        predictions_cpu = predictions.cpu().numpy()
        y_test_cpu = self.y_test.cpu().numpy()

        predictions_original = self.data_processor.inverse_transform_target(predictions_cpu)
        y_test_original = self.data_processor.inverse_transform_target(y_test_cpu)

        mse = mean_squared_error(y_test_original, predictions_original)
        mae = mean_absolute_error(y_test_original, predictions_original)
        rmse = np.sqrt(mse)

        print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        results = {'mse': mse, 'mae': mae, 'rmse': rmse}
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
        """Plot the results."""
        print("Plotting results...")
        
        # Plot training loss
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot prediction results (first few samples)
        plt.subplot(2, 2, 2)
        # Ensure predictions and actuals are 2D
        if predictions.ndim > 2: predictions = predictions.squeeze()
        if actuals.ndim > 2: actuals = actuals.squeeze()
        
        for i in range(min(5, len(predictions))):
            plt.plot(actuals[i], label=f'Actual {i+1}', alpha=0.7)
            plt.plot(predictions[i], label=f'Predicted {i+1}', alpha=0.7, linestyle='--')
        plt.title('Prediction vs Actual (First 5 samples)')
        plt.xlabel('Time Steps')
        plt.ylabel('Power')
        plt.legend()
        plt.grid(True)
        
        # Plot scatter plot
        plt.subplot(2, 2, 3)
        plt.scatter(actuals.flatten(), predictions.flatten(), alpha=0.5)
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True)
        
        # Plot error distribution
        plt.subplot(2, 2, 4)
        errors = predictions.flatten() - actuals.flatten()
        plt.hist(errors, bins=50, alpha=0.7)
        plt.title('Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'results.png'), dpi=300, bbox_inches='tight')
        plt.close()


def run_experiment(config):
    """Run a single experiment."""
    trainer = LSTMTrainer(config)
    trainer.prepare_data()
    trainer.build_model()
    trainer.train()
    predictions, actuals = trainer.evaluate()
    trainer.plot_results(predictions, actuals)
    # Plot gradient flow after training is complete
    plot_gradient_flow(trainer.gradient_history, os.path.join(trainer.results_dir, 'gradient_flow.png'))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='LSTM Model Training Script')
    parser.add_argument('--task', type=str, default='all', choices=['short', 'long', 'all'])
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--model_size', type=str, default='standard', choices=['standard', 'large'])
    args = parser.parse_args()

    def run_task(task_name):
        print(f"\n{'='*50}\nRunning LSTM - {task_name.replace('_', ' ').title()} ({args.model_size} model)\n{'='*50}")
        config = get_config('lstm', task_name.split('_')[0], model_size=args.model_size)
        if args.epochs: config['epochs'] = args.epochs
        
        if args.results_dir:
            config['results_dir'] = args.results_dir
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_size_str = f"_{args.model_size}" if args.model_size != 'standard' else ""
            config['results_dir'] = os.path.join('results', f"lstm_{task_name}{model_size_str}_{timestamp}")
        
        run_experiment(config)

    if args.task == 'all':
        run_task('short_term')
        run_task('long_term')
    else:
        run_task(f"{args.task}_term")


if __name__ == "__main__":
    main()

