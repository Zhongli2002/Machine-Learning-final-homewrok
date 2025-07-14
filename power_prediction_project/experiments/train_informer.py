"""
Informer Model Training Script
Used to train the Informer model for time series power prediction.
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
from tqdm import tqdm
import warnings
import argparse
from datetime import datetime
warnings.filterwarnings('ignore')

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import PowerDataProcessor, create_data_loaders
from models.informer_model import create_informer_model
from config import get_config, get_device_config
from utils.training_utils import log_gradients, plot_gradient_flow


class InformerTrainer:
    """Informer Trainer"""

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

        # Placeholder attributes
        self.model: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.criterion = nn.MSELoss()
        self.train_losses, self.val_losses = [], []
        self.gradient_history = {}

    # -------------- Data ------------------
    def prepare_data(self):
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

        self.data_processor.fit_scalers(train_df)
        train_data = self.data_processor.transform_data(train_df)
        test_data = self.data_processor.transform_data(test_df)

        X_train, y_train = self.data_processor.create_sequences(
            train_data, self.config['sequence_length'], self.config['prediction_length']
        )
        X_test, y_test = self.data_processor.create_sequences(
            test_data, self.config['sequence_length'], self.config['prediction_length']
        )

        val_size = int(len(X_train) * self.config['val_split'])
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_train, y_train = X_train[:-val_size], y_train[:-val_size]

        self.train_loader, self.val_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size=self.config['batch_size']
        )
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        self.y_test = torch.FloatTensor(y_test).to(self.device)

    # -------------- Model ------------------
    def build_model(self):
        if self.data_processor.feature_columns is None:
            raise ValueError("Call prepare_data first")
        input_size = len(self.data_processor.feature_columns) + 1

        self.model = create_informer_model(
            input_size=input_size,
            output_length=self.config['prediction_length'],
            d_model=self.config['d_model'],
            n_head=self.config['nhead'],
            num_layers=self.config['num_layers'],
            d_ff=self.config['d_ff'],
            dropout=self.config['dropout'],
            distil=self.config['distil']
        ).to(self.device)

        if self.use_multi_gpu and len(self.gpu_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Informer parameters: {total_params:,}")

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config['epochs'], eta_min=1e-6
        )

    # -------------- Train / Validate ------------------
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in self.train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            log_gradients(self.model, epoch, self.gradient_history)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self):
        best_val = float('inf'); patience_counter = 0
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step()
            if val_loss < best_val:
                best_val = val_loss; patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.results_dir, 'best_model.pth'))
            else:
                patience_counter += 1
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config['epochs']}  Train {train_loss:.4f}  Val {val_loss:.4f}")
            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch+1}"); break
        self.model.load_state_dict(torch.load(os.path.join(self.results_dir, 'best_model.pth')))
        print("Informer training run complete.")
        # Plot gradient flow
        plot_gradient_flow(self.gradient_history, os.path.join(self.results_dir, 'gradient_flow.png'))

    # -------------- Evaluate / Plot ------------------
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
        # Implementation of plot_results method
        pass

    # -------------- Run ------------------
    def run(self):
        self.prepare_data(); self.build_model(); self.train(); self.evaluate()


def run_experiment(config):
    """Run a single experiment."""
    trainer = InformerTrainer(config)
    trainer.prepare_data()
    trainer.build_model()
    trainer.train()
    predictions, actuals = trainer.evaluate()
    trainer.plot_results(predictions, actuals)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Informer Model Training')
    parser.add_argument('--task', type=str, default='short', choices=['short', 'long'], help='Task type')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--results_dir', type=str, default=None, help='Results directory')
    parser.add_argument('--model_size', type=str, default='standard', choices=['standard', 'large'], help='Model size')
    args = parser.parse_args()

    print(f"Running Informer model, Task: {args.task}, Model Size: {args.model_size}")

    config = get_config('informer', args.task, model_size=args.model_size)
    
    if args.epochs:
        config['epochs'] = args.epochs
    
    if args.results_dir:
        config['results_dir'] = args.results_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_size_str = f"_{args.model_size}" if args.model_size == 'large' else ""
        config['results_dir'] = f"results/informer_{args.task}{model_size_str}_{timestamp}"

    run_experiment(config)


if __name__ == "__main__":
    main() 