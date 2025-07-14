

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MultiScaleConv1D(nn.Module):
    
    def __init__(self, input_channels, output_channels, kernel_sizes=[3, 5, 7], dropout=0.1):
        super(MultiScaleConv1D, self).__init__()
        
        self.convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv = nn.Sequential(
                nn.Conv1d(input_channels, output_channels // len(kernel_sizes), 
                         kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(output_channels // len(kernel_sizes)),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.convs.append(conv)
        
        # 1x1 convolution for dimensionality reduction
        self.pointwise_conv = nn.Conv1d(input_channels, output_channels // len(kernel_sizes), 1)
        
        # Output projection
        total_channels = output_channels // len(kernel_sizes) * (len(kernel_sizes) + 1)
        self.output_proj = nn.Conv1d(total_channels, output_channels, 1)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, channels, seq_len].
            
        Returns:
            torch.Tensor: Output tensor [batch_size, output_channels, seq_len].
        """
        # Multi-scale convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))
        
        # 1x1 convolution
        pointwise_output = self.pointwise_conv(x)
        conv_outputs.append(pointwise_output)
        
        # Concatenate all outputs
        concat_output = torch.cat(conv_outputs, dim=1)
        
        # Output projection
        output = self.output_proj(concat_output)
        
        return output


class AdaptiveAttention(nn.Module):
    """Adaptive Attention Mechanism"""
    
    def __init__(self, input_dim, hidden_dim=None):
        """
        Initialize the adaptive attention.
        
        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
        """
        super(AdaptiveAttention, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.scale = math.sqrt(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, input_dim].
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, input_dim].
        """
        batch_size, seq_len, input_dim = x.size()
        
        # Calculate query, key, value
        Q = self.query(x)  # [batch_size, seq_len, hidden_dim]
        K = self.key(x)    # [batch_size, seq_len, hidden_dim]
        V = self.value(x)  # [batch_size, seq_len, input_dim]
        
        # Calculate attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights
        output = torch.matmul(attention_weights, V)
        
        return output


class ConvLSTMTransformer(nn.Module):
    """Innovative Hybrid Model: ConvLSTMTransformer"""
    
    def __init__(self, input_size, conv_channels=64, lstm_hidden=128, 
                 transformer_dim=256, nhead=8, num_transformer_layers=4,
                 output_size=90, dropout=0.1):
        """
        Initialize the ConvLSTMTransformer model.
        
        Args:
            input_size (int): Dimension of input features.
            conv_channels (int): Number of convolutional channels.
            lstm_hidden (int): Dimension of LSTM hidden layers.
            transformer_dim (int): Dimension of the Transformer.
            nhead (int): Number of attention heads.
            num_transformer_layers (int): Number of Transformer layers.
            output_size (int): Length of the output sequence.
            dropout (float): Dropout probability.
        """
        super(ConvLSTMTransformer, self).__init__()
        
        self.input_size = input_size
        self.conv_channels = conv_channels
        self.lstm_hidden = lstm_hidden
        self.transformer_dim = transformer_dim
        self.output_size = output_size
        
        # 1. Multi-scale convolutional feature extraction
        self.conv_feature_extractor = nn.Sequential(
            MultiScaleConv1D(input_size, conv_channels, [3, 5, 7], dropout),
            MultiScaleConv1D(conv_channels, conv_channels, [3, 5, 7], dropout)
        )
        
        # 2. Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # 3. Feature fusion layer
        self.feature_fusion = nn.Linear(lstm_hidden * 2, transformer_dim)
        
        # 4. Adaptive attention
        self.adaptive_attention = AdaptiveAttention(transformer_dim)
        
        # 5. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_transformer_layers)
        
        # 6. Multi-head predictor
        short_term_size = output_size // 2
        long_term_size = output_size - short_term_size  # Ensure the sum equals output_size
        
        self.short_term_predictor = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, short_term_size)
        )
        
        self.long_term_predictor = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, long_term_size)
        )
        
        # 7. Output fusion layer
        combined_size = short_term_size + long_term_size
        self.output_fusion = nn.Linear(combined_size, output_size)
        
        # 8. Residual connections and layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(conv_channels),
            nn.LayerNorm(transformer_dim),
            nn.LayerNorm(transformer_dim)
        ])
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, input_size].
            
        Returns:
            torch.Tensor: Output tensor [batch_size, output_size].
        """
        batch_size, seq_len, input_size = x.size()
        
        # 1. Multi-scale convolutional feature extraction
        # Transpose dimensions for convolution: [batch_size, input_size, seq_len]
        x_conv = x.transpose(1, 2)
        conv_features = self.conv_feature_extractor(x_conv)
        conv_features = self.layer_norms[0](conv_features.transpose(1, 2))  # [batch_size, seq_len, conv_channels]
        
        # Residual connection (if dimensions match)
        if conv_features.size(-1) == x.size(-1):
            conv_features = conv_features + x
        
        # 2. Bidirectional LSTM for temporal modeling
        lstm_output, (hn, cn) = self.lstm(conv_features)
        
        # 3. Feature fusion
        fused_features = self.feature_fusion(lstm_output)
        fused_features = self.layer_norms[1](fused_features)
        
        # 4. Adaptive attention
        attention_output = self.adaptive_attention(fused_features)
        attention_output = self.layer_norms[2](attention_output + fused_features)  # Residual connection
        
        # 5. Transformer encoding
        transformer_output = self.transformer_encoder(attention_output)
        
        # 6. Global feature aggregation (using attention-weighted average)
        # Calculate importance weights for each time step
        importance_weights = torch.softmax(
            torch.mean(transformer_output, dim=-1, keepdim=True), dim=1
        )
        global_features = torch.sum(transformer_output * importance_weights, dim=1)
        
        # 7. Multi-head prediction
        short_term_pred = self.short_term_predictor(global_features)
        long_term_pred = self.long_term_predictor(global_features)
        
        # 8. Fuse prediction results
        combined_pred = torch.cat([short_term_pred, long_term_pred], dim=1)
        final_output = self.output_fusion(combined_pred)
        
        return final_output


class WaveletTransformerModel(nn.Module):
    """Wavelet Transform-Transformer Hybrid Model"""
    
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=6, 
                 output_size=90, dropout=0.1):
        """
        Initialize the Wavelet Transform-Transformer model.
        
        Args:
            input_size (int): Dimension of input features.
            d_model (int): The dimension of the model.
            nhead (int): Number of attention heads.
            num_layers (int): Number of layers.
            output_size (int): Length of the output sequence.
            dropout (float): Dropout probability.
        """
        super(WaveletTransformerModel, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        
        # Wavelet transform simulation (using convolutions)
        self.wavelet_conv = nn.ModuleList([
            nn.Conv1d(input_size, d_model // 4, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]  # Different scales of wavelets
        ])
        
        # Input projection
        self.input_projection = nn.Linear(d_model, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, input_size].
            
        Returns:
            torch.Tensor: Output tensor [batch_size, output_size].
        """
        batch_size, seq_len, input_size = x.size()
        
        # Wavelet transform feature extraction
        x_conv = x.transpose(1, 2)  # [batch_size, input_size, seq_len]
        wavelet_features = []
        
        for conv in self.wavelet_conv:
            feature = conv(x_conv)  # [batch_size, d_model//4, seq_len]
            wavelet_features.append(feature)
        
        # Concatenate wavelet features
        wavelet_output = torch.cat(wavelet_features, dim=1)  # [batch_size, d_model, seq_len]
        wavelet_output = wavelet_output.transpose(1, 2)  # [batch_size, seq_len, d_model]
        
        # Input projection and positional encoding
        x = self.input_projection(wavelet_output)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer encoding
        transformer_output = self.transformer(x)
        
        # Global average pooling
        global_features = transformer_output.mean(dim=1)
        
        # Output prediction
        output = self.output_layer(global_features)
        
        return output


def create_innovative_model(model_type='conv_lstm_transformer', input_size=13, 
                          output_size=90, **kwargs):
    """
    Create an innovative model.
    
    Args:
        model_type (str): The type of model.
        input_size (int): Dimension of input features.
        output_size (int): Length of the output sequence.
        **kwargs: Other parameters.
        
    Returns:
        nn.Module: The innovative model.
    """
    if model_type == 'conv_lstm_transformer':
        return ConvLSTMTransformer(
            input_size=input_size,
            conv_channels=kwargs.get('conv_channels', 64),
            lstm_hidden=kwargs.get('lstm_hidden', 128),
            transformer_dim=kwargs.get('transformer_dim', 256),
            nhead=kwargs.get('nhead', 8),
            num_transformer_layers=kwargs.get('num_transformer_layers', 4),
            output_size=output_size,
            dropout=kwargs.get('dropout', 0.1)
        )
    elif model_type == 'wavelet_transformer':
        return WaveletTransformerModel(
            input_size=input_size,
            d_model=kwargs.get('d_model', 256),
            nhead=kwargs.get('nhead', 8),
            num_layers=kwargs.get('num_layers', 6),
            output_size=output_size,
            dropout=kwargs.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    batch_size = 16
    seq_len = 90
    input_size = 13
    output_size = 90
    
    x = torch.randn(batch_size, seq_len, input_size).to(device)
    
    # Test ConvLSTMTransformer model
    model = create_innovative_model('conv_lstm_transformer', 
                                  input_size=input_size, 
                                  output_size=output_size).to(device)
    output = model(x)
    print(f"ConvLSTMTransformer output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test WaveletTransformer model
    model = create_innovative_model('wavelet_transformer', 
                                  input_size=input_size, 
                                  output_size=output_size).to(device)
    output = model(x)
    print(f"WaveletTransformer output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

