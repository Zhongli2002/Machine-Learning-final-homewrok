
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.enhanced_lstm import create_enhanced_lstm_model
from models.enhanced_transformer import create_enhanced_transformer_model


class GatingNetwork(nn.Module):
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_experts: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_experts)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, features]
        # Use mean pooling to get global features
        x_pooled = x.mean(dim=1)  # [batch, features]
        
        h = F.relu(self.fc1(x_pooled))
        h = self.dropout(h)
        gate_weights = self.fc2(h)  # [batch, num_experts]
        gate_weights = F.softmax(gate_weights, dim=-1)
        
        return gate_weights


class MoEModel(nn.Module):
    """Mixture of Experts model combining LSTM and Transformer experts."""
    
    def __init__(
        self,
        input_size: int,
        output_length: int,
        # LSTM expert params
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.2,
        # Transformer expert params
        transformer_d_model: int = 256,
        transformer_nhead: int = 8,
        transformer_num_layers: int = 4,
        transformer_dropout: float = 0.1,
        # Gating params
        gate_hidden_size: int = 128,
        # Task-aware routing
        task_embedding_dim: int = 16,
    ):
        super().__init__()
        
        # Create experts
        self.lstm_expert = create_enhanced_lstm_model(
            input_size=input_size,
            output_length=output_length,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            bidirectional=True
        )
        
        self.transformer_expert = create_enhanced_transformer_model(
            input_size=input_size,
            output_length=output_length,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            dropout=transformer_dropout
        )
        
        # Gating network
        self.gating = GatingNetwork(input_size, gate_hidden_size, num_experts=2)
        
        # Task embedding (short vs long term)
        self.task_embedding = nn.Embedding(2, task_embedding_dim)  # 0: short, 1: long
        self.task_proj = nn.Linear(task_embedding_dim, input_size)
        
        # Optional: learnable expert combination
        self.expert_weights = nn.Parameter(torch.ones(2) / 2)
        
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None, return_gate_weights: bool = False):
        """
        Forward pass through MoE.
        
        Args:
            x: Input tensor [batch, seq_len, features]
            task_id: Optional task identifier (0: short-term, 1: long-term)
            return_gate_weights: If True, returns gate weights along with output.
        
        Returns:
            Output predictions [batch, output_length]
            or (predictions, gate_weights) if return_gate_weights is True.
        """
        batch_size = x.size(0)
        
        # Add task information if provided
        if task_id is not None:
            task_emb = self.task_embedding(torch.tensor([task_id]).to(x.device))
            task_emb = task_emb.expand(batch_size, 1, -1)  # [batch, 1, task_dim]
            task_features = self.task_proj(task_emb)  # [batch, 1, input_size]
            # Add task features to first timestep
            x = x.clone()
            x[:, 0, :] = x[:, 0, :] + task_features.squeeze(1)
        
        # Get expert outputs
        lstm_out = self.lstm_expert(x)  # [batch, output_length]
        transformer_out = self.transformer_expert(x)  # [batch, output_length]
        
        # Get gating weights
        gate_weights = self.gating(x)  # [batch, 2]
        
        # Combine expert outputs
        expert_outputs = torch.stack([lstm_out, transformer_out], dim=1)  # [batch, 2, output_length]
        g_weights_expanded = gate_weights.unsqueeze(-1)  # [batch, 2, 1]
        
        # Weighted sum
        output = (expert_outputs * g_weights_expanded).sum(dim=1)  # [batch, output_length]
        
        if return_gate_weights:
            return output, gate_weights
            
        return output
    

def create_moe_model(
    input_size: int,
    output_length: int,
    lstm_hidden_size: int = 128,
    lstm_num_layers: int = 2,
    lstm_dropout: float = 0.2,
    transformer_d_model: int = 256,
    transformer_nhead: int = 8,
    transformer_num_layers: int = 4,
    transformer_dropout: float = 0.1,
    gate_hidden_size: int = 128,
):
    """Factory function to create MoE model."""
    return MoEModel(
        input_size=input_size,
        output_length=output_length,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=lstm_dropout,
        transformer_d_model=transformer_d_model,
        transformer_nhead=transformer_nhead,
        transformer_num_layers=transformer_num_layers,
        transformer_dropout=transformer_dropout,
        gate_hidden_size=gate_hidden_size,
    )


if __name__ == "__main__":
    # Quick test
    batch_size = 4
    seq_len = 90
    input_features = 16
    output_len = 90
    
    model = create_moe_model(input_features, output_len)
    x = torch.randn(batch_size, seq_len, input_features)
    
    # Test without task id
    out = model(x)
    print(f"Output shape: {out.shape}")  # Expected: [4, 90]
    
    # Test with task id
    out_short = model(x, task_id=0)
    out_long = model(x, task_id=1)
    print(f"Short-term output shape: {out_short.shape}")
    print(f"Long-term output shape: {out_long.shape}")
    
    # Check gating weights
    out_with_weights, weights = model(x, return_gate_weights=True)
    print(f"Output shape (with weights): {out_with_weights.shape}")
    print(f"Gating weights shape: {weights.shape}")
    print(f"Gating weights sample: {weights[0].tolist()}") # Expected: [w1, w2] summing to 1 