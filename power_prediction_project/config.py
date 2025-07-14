"""
Project Configuration File
Contains default configuration parameters for all models.
"""

import os

# --- Path Configuration ---
# Get the directory where config.py is located, which is the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# --- Data File Paths ---
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test.csv')

# --- Common Configuration - Shared by all models ---
COMMON_CONFIG = {
    'train_path': TRAIN_DATA_PATH,
    'test_path': TEST_DATA_PATH,
    'scaler_type': 'standard',  # 'standard' or 'minmax'
    'sequence_length': 90,
    'val_split': 0.2,
    'random_seed': 42,
    'device': 'auto',  # 'auto', 'cpu', 'cuda'
    'use_multi_gpu': True,   # Enable multi-GPU for comprehensive experiments
    'gpu_ids': [0, 1],       # Specify which GPU IDs to use
    'lr_patience': 200,      # Patience for learning rate scheduler
}

# --- LSTM Model Base Configuration ---
LSTM_BASE_CONFIG = {
    **COMMON_CONFIG,
    'model_type': 'basic',  # 'basic', 'bidirectional', 'attention'
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.25,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 5e-5,
    'epochs': 300,
    'patience': 200,
}

# LSTM Large Model Configuration is now split into short and long configurations
# See LSTM_LARGE_SHORT_CONFIG and LSTM_LARGE_LONG_CONFIG below

# --- LSTM Short-Term Prediction Configuration ---
LSTM_SHORT_CONFIG = {
    **LSTM_BASE_CONFIG,
    'prediction_length': 90,
    'results_dir': os.path.join(RESULTS_DIR, 'lstm_short_term'),
    # Configurations optimized for 568 samples
    'dropout': 0.2,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'epochs': 200,
    'patience': 200,
}

# --- LSTM Long-Term Prediction Configuration ---
LSTM_LONG_CONFIG = {
    **LSTM_BASE_CONFIG,
    'prediction_length': 365,
    'results_dir': os.path.join(RESULTS_DIR, 'lstm_long_term'),
    # Configurations optimized for 293 samples
    'hidden_size': 128,
    'num_layers': 3,
    'dropout': 0.3,
    'batch_size': 16,
    'learning_rate': 0.0005,
    'weight_decay': 1e-3,
    'epochs': 150,
    'patience': 200,
}

# --- Transformer Model Base Configuration ---
TRANSFORMER_BASE_CONFIG = {
    **COMMON_CONFIG,
    'model_type': 'encoder',  # 'encoder', 'decoder', 'attention'
    'd_model': 64,
    'nhead': 8,
    'num_layers': 4,
    'dim_feedforward': 256,
    'dropout': 0.25,
    'batch_size': 32,
    'learning_rate': 0.0005,
    'weight_decay': 5e-5,
    'epochs': 200,
    'patience': 200,
}

# --- Transformer Short-Term Prediction Configuration ---
TRANSFORMER_SHORT_CONFIG = {
    **TRANSFORMER_BASE_CONFIG,
    'prediction_length': 90,
    'results_dir': os.path.join(RESULTS_DIR, 'transformer_short_term'),
    # Optimized for 568 samples
    'batch_size': 32,
    'learning_rate': 0.0005,
    'epochs': 150,
    'patience': 200,
}

# --- Transformer Long-Term Prediction Configuration ---
TRANSFORMER_LONG_CONFIG = {
    **TRANSFORMER_BASE_CONFIG,
    'prediction_length': 365,
    'results_dir': os.path.join(RESULTS_DIR, 'transformer_long_term'),
    # Optimized for 293 samples
    'd_model': 256,
    'num_layers': 8,
    'dim_feedforward': 1024,
    'dropout': 0.2,
    'batch_size': 16,
    'learning_rate': 0.0003,
    'epochs': 100,
    'patience': 200,
}

# --- Innovative Model Base Configuration ---
INNOVATIVE_BASE_CONFIG = {
    **COMMON_CONFIG,
    'model_type': 'conv_lstm_transformer',  # 'conv_lstm_transformer', 'wavelet_transformer'
    'conv_channels': 64,
    'lstm_hidden': 128,
    'transformer_dim': 128,
    'nhead': 8,
    'num_transformer_layers': 2,
    'dropout': 0.3,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'weight_decay': 1e-4,
    'epochs': 150,
    'patience': 200,
}

# --- Innovative Model Short-Term Prediction Configuration ---
INNOVATIVE_SHORT_CONFIG = {
    **INNOVATIVE_BASE_CONFIG,
    'prediction_length': 90,
    'results_dir': os.path.join(RESULTS_DIR, 'innovative_short_term'),
    # Optimized for 568 samples
    'batch_size': 32,
    'learning_rate': 0.0005,
    'dropout': 0.2,
    'epochs': 120,
    'patience': 200,
}

# --- Innovative Model Long-Term Prediction Configuration ---
INNOVATIVE_LONG_CONFIG = {
    **INNOVATIVE_BASE_CONFIG,
    'prediction_length': 365,
    'results_dir': os.path.join(RESULTS_DIR, 'innovative_long_term'),
    # Optimized for 293 samples
    'conv_channels': 128,
    'lstm_hidden': 256,
    'transformer_dim': 512,
    'num_transformer_layers': 6,
    'dropout': 0.3,
    'batch_size': 16,
    'learning_rate': 0.0003,
    'epochs': 100,
    'patience': 200,
}

# --- Informer Model Base Configuration ---
INFORMER_BASE_CONFIG = {
    **COMMON_CONFIG,
    'd_model': 64,
    'nhead': 8,
    'num_layers': 2,
    'e_layers': 2,
    'd_layers': 1,
    'd_ff': 256,
    'dropout': 0.25,
    'distil': True,
    'batch_size': 32,
    'learning_rate': 0.0005,
    'weight_decay': 5e-5,
    'epochs': 100,
    'patience': 200,
}

INFORMER_SHORT_CONFIG = {
    **INFORMER_BASE_CONFIG,
    'prediction_length': 90,
    'results_dir': os.path.join(RESULTS_DIR, 'informer_short_term'),
    'batch_size': 32,
    'learning_rate': 0.0007,
    'epochs': 120,
    'patience': 200,
}

INFORMER_LONG_CONFIG = {
    **INFORMER_BASE_CONFIG,
    'prediction_length': 365,
    'results_dir': os.path.join(RESULTS_DIR, 'informer_long_term'),
    'd_model': 256,
    'num_layers': 6,
    'd_ff': 1024,
    'dropout': 0.3,
    'batch_size': 16,
    'learning_rate': 0.0004,
    'epochs': 100,
    'patience': 200,
}

# --- MoE Model Base Configuration ---
MOE_BASE_CONFIG = {
    **COMMON_CONFIG,
    'num_experts': 4,
    'top_k': 2,
    'expert_hidden_size': 64,
    'gating_hidden_size': 128,
    'dropout': 0.3,
    'lstm_hidden_size': 128,
    'lstm_num_layers': 2,
    'lstm_dropout': 0.2,
    'transformer_d_model': 128,
    'transformer_nhead': 8,
    'transformer_num_layers': 4,
    'transformer_dropout': 0.1,
    'gate_hidden_size': 64,
    'batch_size': 32,
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,
    'epochs': 150,
    'patience': 200,
}

# --- MoE Short-Term Configuration ---
MOE_SHORT_CONFIG = {
    **MOE_BASE_CONFIG,
    'prediction_length': 90,
    'results_dir': os.path.join(RESULTS_DIR, 'moe_short_term'),
    'batch_size': 32,
    'learning_rate': 0.0008,
    'epochs': 150,
    'patience': 200,
}

# --- MoE Long-Term Configuration ---
MOE_LONG_CONFIG = {
    **MOE_BASE_CONFIG,
    'prediction_length': 365,
    'results_dir': os.path.join(RESULTS_DIR, 'moe_long_term'),
    'lstm_hidden_size': 256,
    'lstm_num_layers': 3,
    'transformer_d_model': 384,
    'transformer_num_layers': 6,
    'batch_size': 16,
    'learning_rate': 0.0004,
    'epochs': 120,
    'patience': 200,
}

# --- Enhanced LSTM Model Configuration ---
ENHANCED_LSTM_BASE_CONFIG = {
    **COMMON_CONFIG,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.25,
    'bidirectional': True,
    'attention_size': 64,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 5e-5,
    'epochs': 150,
    'patience': 200,
}

ENHANCED_LSTM_SHORT_CONFIG = {
    **ENHANCED_LSTM_BASE_CONFIG,
    'prediction_length': 90,
    'results_dir': os.path.join(RESULTS_DIR, 'enhanced_lstm_short_term'),
}

ENHANCED_LSTM_LONG_CONFIG = {
    **ENHANCED_LSTM_BASE_CONFIG,
    'prediction_length': 365,
    'results_dir': os.path.join(RESULTS_DIR, 'enhanced_lstm_long_term'),
    'batch_size': 16,
    'learning_rate': 0.0005,
}

# --- Enhanced Transformer Model Configuration ---
ENHANCED_TRANSFORMER_BASE_CONFIG = {
    **COMMON_CONFIG,
    'd_model': 128,
    'nhead': 8,
    'num_layers': 4,
    'dropout': 0.25,
    'conv_kernel_size': 9,
    'conv_dropout': 0.2,
    'batch_size': 32,
    'learning_rate': 0.0005,
    'weight_decay': 5e-5,
    'epochs': 100,
    'patience': 200,
}

ENHANCED_TRANSFORMER_SHORT_CONFIG = {
    **ENHANCED_TRANSFORMER_BASE_CONFIG,
    'prediction_length': 90,
    'results_dir': os.path.join(RESULTS_DIR, 'enhanced_transformer_short_term'),
}

ENHANCED_TRANSFORMER_LONG_CONFIG = {
    **ENHANCED_TRANSFORMER_BASE_CONFIG,
    'prediction_length': 365,
    'results_dir': os.path.join(RESULTS_DIR, 'enhanced_transformer_long_term'),
    'd_model': 256,
    'batch_size': 16,
}

# --- MoE V2 Model Configuration ---
MOE_V2_BASE_CONFIG = {
    **COMMON_CONFIG,
    # EnhancedLSTM expert params
    'lstm_hidden_size': 128,
    'lstm_num_layers': 2,
    'lstm_dropout': 0.25,
    # EnhancedTransformer expert params
    'transformer_d_model': 128,
    'transformer_nhead': 8,
    'transformer_num_layers': 4,
    'transformer_dropout': 0.2,
    # Gating params
    'gate_hidden_size': 64,
    # Fusion params (new for enhanced MoE V2)
    'fusion_hidden_size': 64,
    # Training params
    'batch_size': 16,
    'learning_rate': 0.0003,
    'weight_decay': 1e-4,
    'epochs': 100,
    'patience': 200,
}

MOE_V2_SHORT_CONFIG = {
    **MOE_V2_BASE_CONFIG,
    'prediction_length': 90,
    'results_dir': os.path.join(RESULTS_DIR, 'moe_v2_short_term'),
    'batch_size': 32,
    'learning_rate': 0.0005,
    'epochs': 80,
}

MOE_V2_LONG_CONFIG = {
    **MOE_V2_BASE_CONFIG,
    'prediction_length': 365,
    'results_dir': os.path.join(RESULTS_DIR, 'moe_v2_long_term'),
    'lstm_hidden_size': 128,
    'transformer_d_model': 256,
    'batch_size': 16,
    'learning_rate': 0.0003,
    'epochs': 120,
}

# --- Evaluation Configuration ---
EVALUATION_CONFIG = {
    'results_dir': os.path.join(RESULTS_DIR, 'evaluation'),
    'num_runs': 5,
    'metrics': ['mse', 'mae', 'rmse'],
    'plot_samples': 5,
}

# --- Model Configuration Mapping ---
MODEL_CONFIGS = {
    'lstm': {
        'short': LSTM_SHORT_CONFIG,
        'long': LSTM_LONG_CONFIG,
    },
    'transformer': {
        'short': TRANSFORMER_SHORT_CONFIG,
        'long': TRANSFORMER_LONG_CONFIG,
    },
    'innovative': {
        'short': INNOVATIVE_SHORT_CONFIG,
        'long': INNOVATIVE_LONG_CONFIG,
    },
    'informer': {
        'short': INFORMER_SHORT_CONFIG,
        'long': INFORMER_LONG_CONFIG,
    },
    'moe': {
        'short': MOE_SHORT_CONFIG,
        'long': MOE_LONG_CONFIG,
    },
    'enhanced_lstm': {
        'short': ENHANCED_LSTM_SHORT_CONFIG,
        'long': ENHANCED_LSTM_LONG_CONFIG,
    },
    'enhanced_transformer': {
        'short': ENHANCED_TRANSFORMER_SHORT_CONFIG,
        'long': ENHANCED_TRANSFORMER_LONG_CONFIG,
    },
    'moe_v2': {
        'short': MOE_V2_SHORT_CONFIG,
        'long': MOE_V2_LONG_CONFIG,
    }
}

# --- Large Model Configurations (for parameter scaling experiments) ---

# Large LSTM configurations
LSTM_LARGE_SHORT_CONFIG = {
    **LSTM_SHORT_CONFIG,
    'hidden_size': 256,
    'num_layers': 4,
    'dropout': 0.3,
    'batch_size': 64,
    'learning_rate': 0.0008,
    'weight_decay': 1e-4,
    'results_dir': os.path.join(RESULTS_DIR, 'lstm_large_short_term'),
}

LSTM_LARGE_LONG_CONFIG = {
    **LSTM_LONG_CONFIG,
    'hidden_size': 512,
    'num_layers': 6,
    'dropout': 0.4,
    'batch_size': 32,
    'learning_rate': 0.0005,
    'weight_decay': 1e-3,
    'results_dir': os.path.join(RESULTS_DIR, 'lstm_large_long_term'),
}

# Large Transformer configurations
TRANSFORMER_LARGE_SHORT_CONFIG = {
    **TRANSFORMER_SHORT_CONFIG,
    'd_model': 512,
    'nhead': 16,
    'num_layers': 8,
    'dim_feedforward': 2048,
    'dropout': 0.2,
    'batch_size': 32,
    'learning_rate': 0.0003,
    'weight_decay': 1e-4,
    'results_dir': os.path.join(RESULTS_DIR, 'transformer_large_short_term'),
}

TRANSFORMER_LARGE_LONG_CONFIG = {
    **TRANSFORMER_LONG_CONFIG,
    'd_model': 768,
    'nhead': 16,
    'num_layers': 12,
    'dim_feedforward': 3072,
    'dropout': 0.3,
    'batch_size': 16,
    'learning_rate': 0.0002,
    'weight_decay': 1e-4,
    'results_dir': os.path.join(RESULTS_DIR, 'transformer_large_long_term'),
}

# Large Innovative configurations
INNOVATIVE_LARGE_SHORT_CONFIG = {
    **INNOVATIVE_SHORT_CONFIG,
    'conv_channels': 128,
    'lstm_hidden': 256,
    'transformer_dim': 512,
    'nhead': 16,
    'num_transformer_layers': 8,
    'dropout': 0.2,
    'batch_size': 32,
    'learning_rate': 0.0003,
    'weight_decay': 1e-4,
    'results_dir': os.path.join(RESULTS_DIR, 'innovative_large_short_term'),
}

INNOVATIVE_LARGE_LONG_CONFIG = {
    **INNOVATIVE_LONG_CONFIG,
    'conv_channels': 256,
    'lstm_hidden': 512,
    'transformer_dim': 1024,
    'nhead': 16,
    'num_transformer_layers': 12,
    'dropout': 0.3,
    'batch_size': 16,
    'learning_rate': 0.0002,
    'weight_decay': 1e-4,
    'results_dir': os.path.join(RESULTS_DIR, 'innovative_large_long_term'),
}

# Large Informer configurations
INFORMER_LARGE_SHORT_CONFIG = {
    **INFORMER_SHORT_CONFIG,
    'd_model': 512,
    'nhead': 16,
    'num_layers': 8,
    'e_layers': 4,
    'd_layers': 2,
    'd_ff': 2048,
    'dropout': 0.2,
    'batch_size': 32,
    'learning_rate': 0.0003,
    'weight_decay': 1e-4,
    'results_dir': os.path.join(RESULTS_DIR, 'informer_large_short_term'),
}

INFORMER_LARGE_LONG_CONFIG = {
    **INFORMER_LONG_CONFIG,
    'd_model': 768,
    'nhead': 16,
    'num_layers': 12,
    'e_layers': 6,
    'd_layers': 3,
    'd_ff': 3072,
    'dropout': 0.3,
    'batch_size': 16,
    'learning_rate': 0.0002,
    'weight_decay': 1e-4,
    'results_dir': os.path.join(RESULTS_DIR, 'informer_large_long_term'),
}

# Large MoE configurations
MOE_LARGE_SHORT_CONFIG = {
    **MOE_SHORT_CONFIG,
    'lstm_hidden_size': 256,
    'lstm_num_layers': 4,
    'transformer_d_model': 512,
    'transformer_nhead': 16,
    'transformer_num_layers': 8,
    'gate_hidden_size': 128,
    'batch_size': 32,
    'learning_rate': 0.0003,
    'weight_decay': 1e-4,
    'results_dir': os.path.join(RESULTS_DIR, 'moe_large_short_term'),
}

MOE_LARGE_LONG_CONFIG = {
    **MOE_LONG_CONFIG,
    'lstm_hidden_size': 512,
    'lstm_num_layers': 6,
    'transformer_d_model': 768,
    'transformer_nhead': 16,
    'transformer_num_layers': 12,
    'gate_hidden_size': 256,
    'batch_size': 16,
    'learning_rate': 0.0002,
    'weight_decay': 1e-4,
    'results_dir': os.path.join(RESULTS_DIR, 'moe_large_long_term'),
}

# Large Enhanced LSTM configurations
ENHANCED_LSTM_LARGE_SHORT_CONFIG = {
    **ENHANCED_LSTM_SHORT_CONFIG,
    'hidden_size': 256,
    'num_layers': 4,
    'attention_size': 128,
    'dropout': 0.3,
    'batch_size': 32,
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,
    'results_dir': os.path.join(RESULTS_DIR, 'enhanced_lstm_large_short_term'),
}

ENHANCED_LSTM_LARGE_LONG_CONFIG = {
    **ENHANCED_LSTM_LONG_CONFIG,
    'hidden_size': 512,
    'num_layers': 6,
    'attention_size': 256,
    'dropout': 0.4,
    'batch_size': 16,
    'learning_rate': 0.0003,
    'weight_decay': 1e-4,
    'results_dir': os.path.join(RESULTS_DIR, 'enhanced_lstm_large_long_term'),
}

# Large Enhanced Transformer configurations
ENHANCED_TRANSFORMER_LARGE_SHORT_CONFIG = {
    **ENHANCED_TRANSFORMER_SHORT_CONFIG,
    'd_model': 512,
    'nhead': 16,
    'num_layers': 8,
    'conv_kernel_size': 15,
    'dropout': 0.2,
    'batch_size': 32,
    'learning_rate': 0.0003,
    'weight_decay': 1e-4,
    'results_dir': os.path.join(RESULTS_DIR, 'enhanced_transformer_large_short_term'),
}

ENHANCED_TRANSFORMER_LARGE_LONG_CONFIG = {
    **ENHANCED_TRANSFORMER_LONG_CONFIG,
    'd_model': 768,
    'nhead': 16,
    'num_layers': 12,
    'conv_kernel_size': 21,
    'dropout': 0.3,
    'batch_size': 16,
    'learning_rate': 0.0002,
    'weight_decay': 1e-4,
    'results_dir': os.path.join(RESULTS_DIR, 'enhanced_transformer_large_long_term'),
}

# --- MoE V2 Large Model Configurations ---
MOE_V2_LARGE_SHORT_CONFIG = {
    **MOE_V2_SHORT_CONFIG,
    'gating_network_hidden_dim': 64,
    'expert_hidden_dim_transformer': 512,
    'expert_hidden_dim_lstm': 256,
    'batch_size': 32,
    'learning_rate': 0.0003,
    'weight_decay': 1e-4,
    'dropout': 0.2,
    'results_dir': os.path.join(RESULTS_DIR, 'moe_v2_large_short_term'),
}

MOE_V2_LARGE_LONG_CONFIG = {
    **MOE_V2_LONG_CONFIG,
    'gating_network_hidden_dim': 128,
    'expert_hidden_dim_transformer': 1024,
    'expert_hidden_dim_lstm': 512,
    'batch_size': 16,
    'learning_rate': 0.0001,
    'weight_decay': 1e-4,
    'dropout': 0.3,
    'results_dir': os.path.join(RESULTS_DIR, 'moe_v2_large_long_term'),
}

# --- Aggregated Model Configurations ---
MODEL_CONFIGS_LARGE = {
    'lstm': {
        'short': LSTM_LARGE_SHORT_CONFIG,
        'long': LSTM_LARGE_LONG_CONFIG,
    },
    'transformer': {
        'short': TRANSFORMER_LARGE_SHORT_CONFIG,
        'long': TRANSFORMER_LARGE_LONG_CONFIG,
    },
    'innovative': {
        'short': INNOVATIVE_LARGE_SHORT_CONFIG,
        'long': INNOVATIVE_LARGE_LONG_CONFIG,
    },
    'informer': {
        'short': INFORMER_LARGE_SHORT_CONFIG,
        'long': INFORMER_LARGE_LONG_CONFIG,
    },
    'moe': {
        'short': MOE_LARGE_SHORT_CONFIG,
        'long': MOE_LARGE_LONG_CONFIG,
    },
    'enhanced_lstm': {
        'short': ENHANCED_LSTM_LARGE_SHORT_CONFIG,
        'long': ENHANCED_LSTM_LARGE_LONG_CONFIG,
    },
    'enhanced_transformer': {
        'short': ENHANCED_TRANSFORMER_LARGE_SHORT_CONFIG,
        'long': ENHANCED_TRANSFORMER_LARGE_LONG_CONFIG,
    },
    'moe_v2': {
        'short': MOE_V2_LARGE_SHORT_CONFIG,
        'long': MOE_V2_LARGE_LONG_CONFIG,
    }
}

def get_config(model_name, task_type, model_size='standard'):
    """
    Get the configuration for a specific model and task.
    
    Args:
        model_name (str): Model name ('lstm', 'transformer', 'innovative')
        task_type (str): Task type ('short', 'long')
        model_size (str): Model size ('standard', 'large'). Defaults to 'standard'.
        
    Returns:
        dict: Configuration dictionary
    """
    
    config_map = MODEL_CONFIGS if model_size == 'standard' else MODEL_CONFIGS_LARGE
    
    # Handle base model name (e.g., 'lstm' from 'lstm_large')
    base_model_name = model_name.replace('_large', '')

    if base_model_name not in config_map:
        raise ValueError(f"Unknown model: {base_model_name} for size {model_size}")
    
    if task_type not in config_map[base_model_name]:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return config_map[base_model_name][task_type].copy()

def update_config(config, **kwargs):
    """
    Update configuration parameters.
    
    Args:
        config (dict): The original configuration.
        **kwargs: Parameters to update.
        
    Returns:
        dict: The updated configuration.
    """
    updated_config = config.copy()
    updated_config.update(kwargs)
    return updated_config

# --- Device Configuration ---
def get_device_config(use_multi_gpu=True, gpu_ids=None):
    """Get device configuration."""
    import torch
    
    if not torch.cuda.is_available():
        print("CUDA is not available, using CPU.")
        return 'cpu', False, []
    
    available_gpus = torch.cuda.device_count()
    print(f"Detected {available_gpus} GPU(s).")
    
    if use_multi_gpu and available_gpus > 1:
        if gpu_ids is None:
            gpu_ids = list(range(available_gpus))
        else:
            # Validate the specified GPU IDs
            gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < available_gpus]
        
        if len(gpu_ids) > 1:
            print(f"Using multi-GPU: {gpu_ids}")
            for gpu_id in gpu_ids:
                print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            return 'cuda', True, gpu_ids
        else:
            gpu_id = gpu_ids[0] if gpu_ids else 0
            print(f"Only one valid GPU found, using single GPU: {gpu_id}")
            print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            return 'cuda', False, gpu_ids
    else:
        gpu_id = 0
        print(f"Using single GPU: {gpu_id}")
        print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        return 'cuda', False, [gpu_id]

if __name__ == "__main__":
    # Test configuration
    print("Testing configuration file...")
    
    # Test get_config
    lstm_short_config = get_config('lstm', 'short')
    print(f"LSTM short-term config: {lstm_short_config}")
    
    # Test update_config
    updated_config = update_config(lstm_short_config, learning_rate=0.01, epochs=100)
    print(f"Updated config: learning_rate={updated_config['learning_rate']}, epochs={updated_config['epochs']}")
    
    # Test device configuration
    device, use_multi_gpu, gpu_ids = get_device_config()
    print(f"Device: {device}, Multi-GPU: {use_multi_gpu}, GPU IDs: {gpu_ids}")
    
    print("Configuration file test complete!")

# --- Parameter Scaling Analysis Configuration ---
PARAMETER_SCALING_CONFIG = {
    'results_dir': os.path.join(RESULTS_DIR, 'parameter_scaling_analysis'),
    'model_sizes': ['small', 'medium', 'large'],
    'size_mapping': {
        'small': {
            'lstm_hidden': 64,
            'transformer_d_model': 128,
            'num_layers': 2,
        },
        'medium': {
            'lstm_hidden': 128,
            'transformer_d_model': 256,
            'num_layers': 4,
        },
        'large': {
            'lstm_hidden': 256,
            'transformer_d_model': 512,
            'num_layers': 8,
        }
    }
}

# --- GPU Memory and Performance Configuration ---
GPU_PERFORMANCE_CONFIG = {
    'memory_efficient': {
        'gradient_checkpointing': True,
        'mixed_precision': True,
        'batch_size_reduction_factor': 0.5,
    },
    'high_performance': {
        'gradient_checkpointing': False,
        'mixed_precision': False,
        'batch_size_increase_factor': 1.5,
    }
}

