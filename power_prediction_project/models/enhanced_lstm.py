
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_length: int = 90,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_length
        
        # Core LSTM - exactly like the original
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Enhancement 1: Simple attention for temporal aggregation
        # Much simpler than before - just learn to weight timesteps
        self.attention = nn.Linear(hidden_size, 1)
        
        # Enhancement 2: Minimal residual connection
        # Project input to hidden size for residual
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Dropout layer (like original)
        self.dropout = nn.Dropout(dropout)
        
        # Output layer (like original)
        self.fc = nn.Linear(hidden_size, output_length)
        
        # Initialize weights like the original LSTM
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights exactly like the original LSTM model."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
        
        torch.nn.init.xavier_uniform_(self.input_projection.weight)
        self.input_projection.bias.data.fill_(0)
        
        torch.nn.init.xavier_uniform_(self.attention.weight)
        self.attention.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Initialize hidden state (exactly like original)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Enhancement 1: Simple attention mechanism
        # Compute attention weights for each timestep
        attn_weights = torch.tanh(self.attention(lstm_out))  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted average of all timesteps
        attended_output = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden_size]
        
        # Enhancement 2: Add simple residual from input
        # Use the last timestep of input as residual
        input_residual = self.input_projection(x[:, -1, :])  # [batch, hidden_size]
        
        # Combine with small residual weight (don't overwhelm the LSTM output)
        combined_output = attended_output + 0.1 * input_residual
        
        # Apply dropout (like original)
        combined_output = self.dropout(combined_output)
        
        # Final output layer (like original)
        output = self.fc(combined_output)
        
        return output


def create_enhanced_lstm_model(
    input_size: int,
    output_length: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False,
):
    return EnhancedLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_length=output_length,
        dropout=dropout,
        bidirectional=bidirectional,
    )

if __name__ == "__main__":
    # Test the enhanced model
    batch, seq_len, feat = 8, 90, 16
    model = create_enhanced_lstm_model(feat, 90)
    
    x = torch.randn(batch, seq_len, feat)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("Enhanced LSTM v3.0 - Minimal but effective enhancements!") 