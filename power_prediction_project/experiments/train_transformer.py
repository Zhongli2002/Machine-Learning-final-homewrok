"""
Transformer Model Training Script
Used to train the Transformer model for time series power prediction.
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
from models.transformer_model import create_transformer_model
from config import get_config, get_device_config
from utils.training_utils import log_gradients, plot_gradient_flow
class TransformerTrainer:
    """Transformer Trainer"""
    
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
        self.device, self.use_multi_gpu, self.gpu_ids = get_device_config(
            config.get('use_multi_gpu', True), 
            config.get('gpu_ids', None)
        )
        
        # Set random seed
        torch.manual_seed(config['random_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config['random_seed'])
        
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
        
        self.X_train = None
        self.y_train = None
        
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
        
        X_train_all, y_train_all = self.data_processor.create_sequences(
            train_data, sequence_length, prediction_length
        )
        X_test, y_test = self.data_processor.create_sequences(
            test_data, sequence_length, prediction_length
        )
        
        print(f"X_train shape: {X_train_all.shape}, y_train shape: {y_train_all.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        # Split into training and validation sets
        val_split = self.config['val_split']
        val_size = int(len(X_train_all) * val_split)
        
        X_val, y_val = X_train_all[-val_size:], y_train_all[-val_size:]
        self.X_train, self.y_train = X_train_all[:-val_size], y_train_all[:-val_size]
        
        # Create data loaders
        self.train_loader, self.val_loader = create_data_loaders(
            self.X_train, self.y_train, X_val, y_val, 
            batch_size=self.config['batch_size']
        )
        
        # Save test data
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        self.y_test = torch.FloatTensor(y_test).to(self.device)
        
        return train_df, test_df
    
    def build_model(self):
        """Build the Transformer model."""
        print("Building Transformer model...")
        
        # Ensure data is prepared
        assert self.X_train is not None and self.y_train is not None, \
            "Data not prepared. Call prepare_data() first."
        
        input_size = self.X_train.shape[2]
        
        # Create model
        self.model = create_transformer_model(
            model_type=self.config.get('model_type', 'encoder'),
            input_size=input_size,
            output_size=self.config['prediction_length'],
            d_model=self.config.get('d_model'),
            nhead=self.config.get('nhead'),
            num_layers=self.config.get('num_layers'),
            dim_feedforward=self.config.get('dim_feedforward'),
            dropout=self.config.get('dropout')
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Multi-GPU support
        if self.use_multi_gpu and len(self.gpu_ids) > 1:
            print(f"Using DataParallel to wrap the model, GPUs: {self.gpu_ids}")
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
        
        # Calculate number of model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config['learning_rate'], 
            weight_decay=self.config.get('weight_decay', 0.0)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', 
            patience=self.config.get('lr_patience', 10), 
            factor=0.5
        )
        
        # Gradient history
        self.gradient_history = {}
        
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
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{self.config['epochs']}, "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                      f"LR: {current_lr:.2e}")
        
        # Load the best model
        assert self.model is not None
        self.model.load_state_dict(torch.load(os.path.join(self.results_dir, 'best_model.pth')))
    
    def evaluate(self):
        """Evaluate the model."""
        print("Evaluating model...")
        
        assert self.model is not None
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            # Batch prediction
            batch_size = 32
            for i in range(0, len(self.X_test), batch_size):
                batch_x = self.X_test[i:i+batch_size]
                batch_y = self.y_test[i:i+batch_size]
                
                outputs = self.model(batch_x)
                
                predictions.append(outputs.cpu().numpy())
                actuals.append(batch_y.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        # Inverse transform
        predictions_orig = []
        actuals_orig = []
        
        for i in range(predictions.shape[0]):
            pred_orig = self.data_processor.inverse_transform_target(predictions[i])
            actual_orig = self.data_processor.inverse_transform_target(actuals[i])
            
            predictions_orig.append(pred_orig)
            actuals_orig.append(actual_orig)
        
        predictions_orig = np.array(predictions_orig)
        actuals_orig = np.array(actuals_orig)
        
        # Calculate metrics
        mse = mean_squared_error(actuals_orig.flatten(), predictions_orig.flatten())
        mae = mean_absolute_error(actuals_orig.flatten(), predictions_orig.flatten())
        rmse = np.sqrt(mse)
        
        print(f"Test MSE: {mse:.6f}")
        print(f"Test MAE: {mae:.6f}")
        print(f"Test RMSE: {rmse:.6f}")
        
        # Save results to a JSON file
        results = {'mse': mse, 'mae': mae, 'rmse': rmse, 'config': self.config}
        
        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        np.savez_compressed(
            os.path.join(self.results_dir, 'predictions.npz'),
            predictions=predictions_orig,
            ground_truth=actuals_orig
        )
        print(f"Predictions and ground truth saved to {self.results_dir}/predictions.npz")
        
        return predictions_orig, actuals_orig
    
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
    """Run a single experiment for the given configuration."""
    trainer = TransformerTrainer(config)
    
    # Prepare data
    trainer.prepare_data()
    
    # Build model
    trainer.build_model()
    
    # Train
    trainer.train()
    
    # Evaluate
    predictions, actuals = trainer.evaluate()
    trainer.plot_results(predictions, actuals)

    # Plot gradient flow
    plot_gradient_flow(trainer.gradient_history, os.path.join(trainer.results_dir, 'gradient_flow.png'))

def main():
    """Main function to run short-term and long-term prediction experiments."""
    parser = argparse.ArgumentParser(description='Transformer Model Training Script')
    parser.add_argument('--task', type=str, default='all', choices=['short', 'long', 'all'],
                        help='Task to run: short-term, long-term, or all.')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Set a random seed for reproducibility.')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Set a custom directory for saving results.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override the number of training epochs.')
    parser.add_argument('--model_size', type=str, default='standard', choices=['standard', 'large'],
                        help='Model size to run (standard or large)')
    args = parser.parse_args()

    if args.task == 'short' or args.task == 'all':
        print("=" * 50)
        print("Transformer Short-term Prediction (90 days)")
        print("=" * 50)
        short_config = get_config('transformer', 'short', model_size=args.model_size)
        if args.random_seed is not None:
            short_config['random_seed'] = args.random_seed
        if args.results_dir is not None:
            short_config['results_dir'] = args.results_dir
        if args.epochs is not None:
            short_config['epochs'] = args.epochs
        run_experiment(short_config)
    
    if args.task == 'long' or args.task == 'all':
        print("\n" + "=" * 50)
        print("Transformer Long-term Prediction (365 days)")
        print("=" * 50)
        long_config = get_config('transformer', 'long', model_size=args.model_size)
        if args.random_seed is not None:
            long_config['random_seed'] = args.random_seed
        if args.results_dir is not None:
            long_config['results_dir'] = args.results_dir
        if args.epochs is not None:
            long_config['epochs'] = args.epochs
        run_experiment(long_config)

if __name__ == "__main__":
    main()