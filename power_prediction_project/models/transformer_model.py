
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional Encoding"""
    
    def __init__(self, d_model, max_len=5000):

        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [seq_len, batch_size, d_model].
            
        Returns:
            torch.Tensor: Tensor with positional encoding added.
        """
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    """Transformer Model Class"""
    
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, 
                 dim_feedforward=2048, dropout=0.1, max_len=5000):
        """
        Initialize the Transformer model.
        
        Args:
            input_size (int): Dimension of input features.
            d_model (int): The dimension of the model.
            nhead (int): Number of attention heads.
            num_layers (int): Number of Transformer layers.
            output_size (int): Length of the output sequence.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout probability.
            max_len (int): The maximum length of the sequence.
        """
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        
        # Input projection layer
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # [seq_len, batch_size, d_model]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.output_projection = nn.Linear(d_model, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.output_projection.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
    
    def forward(self, x, src_mask=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, input_size].
            src_mask (torch.Tensor): Source sequence mask.
            
        Returns:
            torch.Tensor: Output tensor [batch_size, output_size].
        """
        # Transpose dimensions: [batch_size, seq_len, input_size] -> [seq_len, batch_size, input_size]
        x = x.transpose(0, 1)
        
        # Input projection
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Transformer encoding
        output = self.transformer_encoder(x, src_mask)
        
        # Take the output of the last time step
        output = output[-1, :, :]  # [batch_size, d_model]
        
        # Output projection
        output = self.output_projection(output)
        
        return output


class TransformerDecoderModel(nn.Module):
    """Transformer model with a decoder."""
    
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, 
                 num_decoder_layers, output_size, dim_feedforward=2048, 
                 dropout=0.1, max_len=5000):
        """
        Initialize the Transformer model with a decoder.
        
        Args:
            input_size (int): Dimension of input features.
            d_model (int): The dimension of the model.
            nhead (int): Number of attention heads.
            num_encoder_layers (int): Number of encoder layers.
            num_decoder_layers (int): Number of decoder layers.
            output_size (int): Length of the output sequence.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout probability.
            max_len (int): The maximum length of the sequence.
        """
        super(TransformerDecoderModel, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        
        # Input projection layer
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Target projection layer (for decoder input)
        self.target_projection = nn.Linear(1, d_model)  # Predicting only the power value
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        
        # Output layer
        self.output_projection = nn.Linear(d_model, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.target_projection.weight.data.uniform_(-initrange, initrange)
        self.output_projection.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square subsequent mask."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, tgt=None):
        """
        Forward pass.
        
        Args:
            src (torch.Tensor): Source sequence [batch_size, src_len, input_size].
            tgt (torch.Tensor): Target sequence [batch_size, tgt_len, 1].
            
        Returns:
            torch.Tensor: Output tensor [batch_size, output_size].
        """
        batch_size = src.size(0)
        src_len = src.size(1)
        
        # Transpose dimensions
        src = src.transpose(0, 1)  # [src_len, batch_size, input_size]
        
        # Process encoder input
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        
        if tgt is None:
            # Inference mode: auto-regressive generation
            outputs = []
            # Initialize decoder input (using the last power value of the source sequence)
            decoder_input = src[-1:, :, 0:1]  # [1, batch_size, 1]
            
            for i in range(self.output_size):
                # Process target sequence
                tgt_emb = self.target_projection(decoder_input) * math.sqrt(self.d_model)
                tgt_emb = self.pos_encoder(tgt_emb)
                tgt_emb = self.dropout(tgt_emb)
                
                # Generate mask
                tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(0)).to(src.device)
                
                # Transformer forward pass
                output = self.transformer(src, tgt_emb, tgt_mask=tgt_mask)
                
                # Output projection
                pred = self.output_projection(output[-1:, :, :])  # [1, batch_size, 1]
                outputs.append(pred.squeeze(0))  # [batch_size, 1]
                
                # Update decoder input
                decoder_input = torch.cat([decoder_input, pred], dim=0)
            
            return torch.cat(outputs, dim=1)  # [batch_size, output_size]
        
        else:
            # Training mode: use true target sequence
            tgt = tgt.transpose(0, 1)  # [tgt_len, batch_size, 1]
            
            # Process target sequence
            tgt = self.target_projection(tgt) * math.sqrt(self.d_model)
            tgt = self.pos_encoder(tgt)
            tgt = self.dropout(tgt)
            
            # Generate mask
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
            
            # Transformer forward pass
            output = self.transformer(src, tgt, tgt_mask=tgt_mask)
            
            # Output projection
            output = self.output_projection(output)  # [tgt_len, batch_size, 1]
            
            return output.transpose(0, 1).squeeze(-1)  # [batch_size, tgt_len]


class MultiHeadAttentionModel(nn.Module):
    """Multi-Head Attention Model (simplified Transformer)"""
    
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.1):
        """
        Initialize the Multi-Head Attention model.
        
        Args:
            input_size (int): Dimension of input features.
            d_model (int): The dimension of the model.
            nhead (int): Number of attention heads.
            num_layers (int): Number of layers.
            output_size (int): Length of the output sequence.
            dropout (float): Dropout probability.
        """
        super(MultiHeadAttentionModel, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_projection = nn.Linear(d_model, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, input_size].
            
        Returns:
            torch.Tensor: Output tensor [batch_size, output_size].
        """
        # Input projection
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Multi-layer attention
        for attention, layer_norm, feed_forward in zip(
            self.attention_layers, self.layer_norms, self.feed_forwards
        ):
            # Multi-head attention
            attn_output, _ = attention(x, x, x)
            x = layer_norm(x + attn_output)
            
            # Feed-forward network
            ff_output = feed_forward(x)
            x = layer_norm(x + ff_output)
        
        # Global average pooling or take the last time step
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        # Output projection
        output = self.output_projection(x)
        
        return output


def create_transformer_model(model_type='encoder', input_size=13, d_model=128, 
                           nhead=8, num_layers=6, output_size=90, 
                           dim_feedforward=512, dropout=0.1):
    """
    Create a Transformer model.
    
    Args:
        model_type (str): Model type ('encoder', 'decoder', 'attention').
        input_size (int): Dimension of input features.
        d_model (int): The dimension of the model.
        nhead (int): Number of attention heads.
        num_layers (int): Number of layers.
        output_size (int): Length of the output sequence.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout probability.
        
    Returns:
        nn.Module: The Transformer model.
    """
    if model_type == 'encoder':
        return TransformerModel(
            input_size, d_model, nhead, num_layers, output_size, 
            dim_feedforward, dropout
        )
    elif model_type == 'decoder':
        return TransformerDecoderModel(
            input_size, d_model, nhead, num_layers//2, num_layers//2, 
            output_size, dim_feedforward, dropout
        )
    elif model_type == 'attention':
        return MultiHeadAttentionModel(
            input_size, d_model, nhead, num_layers, output_size, dropout
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
    
    # Test Encoder model
    model = create_transformer_model('encoder', input_size=input_size, output_size=output_size).to(device)
    output = model(x)
    print(f"Encoder Transformer output shape: {output.shape}")
    
    # Test Decoder model
    model = create_transformer_model('decoder', input_size=input_size, output_size=output_size).to(device)
    output = model(x)
    print(f"Decoder Transformer output shape: {output.shape}")
    
    # Test Attention model
    model = create_transformer_model('attention', input_size=input_size, output_size=output_size).to(device)
    output = model(x)
    print(f"Attention model output shape: {output.shape}")

