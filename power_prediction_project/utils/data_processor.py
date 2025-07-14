"""
Data Processing Module
Handles cleaning, feature engineering, and time series preparation for the household power consumption dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class PowerDataProcessor:
    """Power Data Processor"""
    
    def __init__(self, scaler_type='standard'):
        """
        Initialize the data processor.
        
        Args:
            scaler_type (str): Type of scaler, 'standard' or 'minmax'.
        """
        self.scaler_type = scaler_type
        self.feature_scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.target_scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.feature_columns = None
        self.target_column = 'Global_active_power'
        
    def load_data(self, train_path, test_path):
        """
        Load training and test data.
        
        Args:
            train_path (str): Path to the training data.
            test_path (str): Path to the test data.
            
        Returns:
            tuple: (train_df, test_df)
        """
        train_df = pd.read_csv(train_path, low_memory=False, encoding='utf-8-sig')
        # Test file has no header, so column names need to be specified manually
        test_df = pd.read_csv(test_path, low_memory=False, encoding='utf-8-sig', header=None)
        # Get column names from training data and apply them to test data
        test_df.columns = train_df.columns

        # Clean column names, removing leading/trailing spaces and special characters
        train_df.columns = train_df.columns.str.strip()
        test_df.columns = test_df.columns.str.strip()
        
        # Convert time column
        train_df['DateTime'] = pd.to_datetime(train_df['DateTime'])
        test_df['DateTime'] = pd.to_datetime(test_df['DateTime'])
        
        return train_df, test_df
    
    def preprocess_data(self, df):
        """
        Preprocess the data.
        
        Args:
            df (pd.DataFrame): The raw dataframe.
            
        Returns:
            pd.DataFrame: The preprocessed dataframe.
        """
        # Copy data to avoid modifying the original dataframe
        df = df.copy()
        
        # Convert all numeric columns to numeric types, setting non-convertible values to NaN
        numeric_cols = [
            'Global_active_power', 'Global_reactive_power', 'Voltage', 
            'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle missing values using modern pandas syntax
        df = df.ffill().bfill()
        
        # Add time-based features
        df['hour'] = df['DateTime'].dt.hour
        df['day'] = df['DateTime'].dt.day
        df['month'] = df['DateTime'].dt.month
        df['weekday'] = df['DateTime'].dt.weekday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        
        # Calculate sub_metering_remainder
        df['sub_metering_remainder'] = (
            df['Global_active_power'] * 1000 / 60 - 
            df['Sub_metering_1'] - df['Sub_metering_2'] - df['Sub_metering_3']
        )
        
        return df
    
    def aggregate_daily_data(self, df):
        """
        Aggregate data by day.
        
        Args:
            df (pd.DataFrame): The minute-level data.
            
        Returns:
            pd.DataFrame: The daily aggregated data.
        """
        df = df.copy()
        df['date'] = df['DateTime'].dt.date
        
        # Aggregate according to requirements
        agg_dict = {
            'Global_active_power': 'sum',
            'Global_reactive_power': 'sum',
            'Sub_metering_1': 'sum',
            'Sub_metering_2': 'sum',
            'Sub_metering_3': 'sum',
            'Voltage': 'mean',
            'Global_intensity': 'mean',
            'RR': 'first',
            'NBJRR1': 'first',
            'NBJRR5': 'first',
            'NBJRR10': 'first',
            'NBJBROU': 'first',
            'sub_metering_remainder': 'sum'
        }
        
        # Add aggregation for time-based features
        if 'hour' in df.columns:
            agg_dict.update({
                'month': 'first',
                'weekday': 'first',
                'is_weekend': 'first'
            })
        
        daily_df = df.groupby('date').agg(agg_dict).reset_index()
        daily_df['DateTime'] = pd.to_datetime(daily_df['date'])
        
        return daily_df
    
    def prepare_features(self, df):
        """
        Prepare feature columns.
        
        Args:
            df (pd.DataFrame): The dataframe.
            
        Returns:
            pd.DataFrame: Dataframe containing the features.
        """
        # Select feature columns
        feature_cols = [
            'Global_reactive_power', 'Voltage', 'Global_intensity',
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
            'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU',
            'sub_metering_remainder'
        ]
        
        # Add time-based features
        if 'month' in df.columns:
            feature_cols.extend(['month', 'weekday', 'is_weekend'])
        
        self.feature_columns = feature_cols
        
        return df[['DateTime', self.target_column] + feature_cols]
    
    def create_sequences(self, data, sequence_length, prediction_length):
        """
        Create time series sequences.
        
        Args:
            data (np.array): Input data.
            sequence_length (int): Length of the input sequence.
            prediction_length (int): Length of the prediction sequence.
            
        Returns:
            tuple: (X, y) Input sequences and target sequences.
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length - prediction_length + 1):
            # Input sequence
            X.append(data[i:(i + sequence_length)])
            # Target sequence
            y.append(data[i + sequence_length:i + sequence_length + prediction_length, 0])  # Only take the target column
        
        return np.array(X), np.array(y)
    
    def fit_scalers(self, train_df):
        """
        Fit the scalers.
        
        Args:
            train_df (pd.DataFrame): The training data.
        """
        # Fit feature scaler
        features = train_df[self.feature_columns].values
        self.feature_scaler.fit(features)
        
        # Fit target scaler
        target = train_df[self.target_column].values.reshape(-1, 1)
        self.target_scaler.fit(target)
    
    def transform_data(self, df):
        """
        Scale the data.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            np.array: The scaled data.
        """
        # Scale features
        features = self.feature_scaler.transform(df[self.feature_columns].values)
        
        # Scale target
        target = self.target_scaler.transform(df[self.target_column].values.reshape(-1, 1))
        
        # Concatenate features and target
        return np.concatenate([target, features], axis=1)
    
    def inverse_transform_target(self, scaled_target):
        """
        Inverse transform the target values.
        
        Args:
            scaled_target (np.array): The scaled target values.
            
        Returns:
            np.array: The target values in their original scale.
        """
        if len(scaled_target.shape) == 1:
            scaled_target = scaled_target.reshape(-1, 1)
        return self.target_scaler.inverse_transform(scaled_target).flatten()


class PowerDataset(Dataset):
    """Power Dataset Class"""
    
    def __init__(self, X, y):
        """
        Initialize the dataset.
        
        Args:
            X (np.array): Input sequences.
            y (np.array): Target sequences.
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create data loaders.
    
    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        batch_size (int): Batch size.
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_dataset = PowerDataset(X_train, y_train)
    val_dataset = PowerDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

