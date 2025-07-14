from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbSparseAttention(nn.Module):

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1, top_u: Optional[int] = None):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.top_u = top_u  # 未使用，后续可扩展

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention: query=key=value=x
        attn_output, _ = self.mha(x, x, x, need_weights=False)
        return self.dropout(attn_output)


class ConvDistillLayer(nn.Module):
    """1-D convolution + ELU + MaxPool to halve sequence length, as in Informer."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, bias=False)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # 长度减半

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model] -> conv requires [batch, d_model, seq_len]
        x = x.transpose(1, 2)
        x = self.pool(self.activation(self.conv(x)))
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """Informer Encoder layer: ProbSparseAttention + ConvDistill (optional) + FFN."""

    def __init__(self, d_model: int, n_head: int, d_ff: int = 256, dropout: float = 0.1, distil: bool = False):
        super().__init__()
        self.self_attn = ProbSparseAttention(d_model, n_head, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.distil = distil
        if distil:
            self.conv_distill = ConvDistillLayer(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self attention + residual
        attn_out = self.self_attn(x)
        x = self.layer_norm1(x + attn_out)

        # Feed forward + residual
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)

        # Distilling convolution (optional)
        if self.distil:
            x = self.conv_distill(x)
        return x


class InformerEncoder(nn.Module):
    """Stack of EncoderLayers with optional distilling between layers."""

    def __init__(self, num_layers: int, d_model: int, n_head: int, d_ff: int, dropout: float, distil: bool = True):
        super().__init__()
        layers = []
        for i in range(num_layers):
            # 在偶数层后做 distilling (i % 2 == 0) 并且 distil 开启
            use_distil = distil and (i % 2 == 0) and (i != num_layers - 1)
            layers.append(EncoderLayer(d_model, n_head, d_ff, dropout, use_distil))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class InformerModel(nn.Module):
    """Informer encoder model for multistep forecasting."""

    def __init__(
        self,
        input_size: int,
        output_length: int,
        d_model: int = 256,
        n_head: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.2,
        distil: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.output_length = output_length

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1024, d_model))  # learnable PE up to 1024

        self.encoder = InformerEncoder(num_layers, d_model, n_head, d_ff, dropout, distil)

        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, output_length)

        nn.init.xavier_uniform_(self.pos_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_size]
        seq_len = x.size(1)
        # Input Projection + positional embedding
        x = self.input_proj(x) + self.pos_embedding[:, :seq_len, :]

        # Encoder
        enc_out = self.encoder(x)

        # Pooling（使用均值）
        enc_out = enc_out.mean(dim=1)  # [batch, d_model]
        enc_out = self.norm(enc_out)

        # Output projection to forecast horizon length
        out = self.projection(enc_out)  # [batch, output_length]
        return out


def create_informer_model(
    input_size: int,
    output_length: int,
    d_model: int = 256,
    n_head: int = 8,
    num_layers: int = 4,
    d_ff: int = 512,
    dropout: float = 0.2,
    distil: bool = True,
):
    """Factory function to create InformerModel with given hyper-params."""
    return InformerModel(
        input_size=input_size,
        output_length=output_length,
        d_model=d_model,
        n_head=n_head,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        distil=distil,
    )


if __name__ == "__main__":
    # Quick sanity test
    batch = 4
    seq_len = 90
    feature_dim = 16
    horizon = 90

    model = create_informer_model(feature_dim, horizon)
    dummy_x = torch.randn(batch, seq_len, feature_dim)
    out = model(dummy_x)
    print(out.shape)  # Expected: [batch, horizon] 