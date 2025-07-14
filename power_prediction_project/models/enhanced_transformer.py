
from __future__ import annotations

import math
import torch
import torch.nn as nn

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(in_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B, L, C] -> conv expects [B, C, L]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x.transpose(1, 2)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EnhancedTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_length: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        # Dilated conv stack (2 layers dilations 1 and 2)
        self.dilated_conv = nn.Sequential(
            DilatedConvBlock(d_model, dilation=1, dropout=dropout),
            DilatedConvBlock(d_model, dilation=2, dropout=dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, output_length)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,L,input]
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.dilated_conv(x)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        pooled = self.norm(pooled)
        return self.fc(pooled)

def create_enhanced_transformer_model(
    input_size:int,
    output_length:int,
    d_model:int=256,
    nhead:int=8,
    num_layers:int=4,
    dim_feedforward:int=512,
    dropout:float=0.1,
):
    return EnhancedTransformer(input_size, output_length, d_model, nhead, num_layers, dim_feedforward, dropout)

if __name__=='__main__':
    m=create_enhanced_transformer_model(16,90)
    print(m(torch.randn(8,90,16)).shape) 