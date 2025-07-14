import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSTMModel(nn.Module):
    """LSTM Model Class"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, output_size]
        """
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Get the output of the last time step
        last_output = lstm_out[:, -1, :]
        
        # Dropout
        last_output = self.dropout(last_output)
        
        # Fully connected layer
        output = self.fc(last_output)
        
        return output


class BidirectionalLSTMModel(nn.Module):
    """Bidirectional LSTM Model Class"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Initialize the Bidirectional LSTM model.
        
        Args:
            input_size (int): Dimension of input features.
            hidden_size (int): Dimension of LSTM hidden layers.
            num_layers (int): Number of LSTM layers.
            output_size (int): Length of the output sequence.
            dropout (float): Dropout probability.
        """
        super(BidirectionalLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer (note that bidirectional LSTM output dim is hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, output_size]
        """
        batch_size = x.size(0)
        
        # Initialize hidden state (bidirectional LSTM requires 2 * num_layers)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Get the output of the last time step
        last_output = lstm_out[:, -1, :]
        
        # Dropout
        last_output = self.dropout(last_output)
        
        # Fully connected layer
        output = self.fc(last_output)
        
        return output


class AttentionLSTMModel(nn.Module):
    """LSTM Model with Attention Mechanism"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Initialize the Attention LSTM model.
        
        Args:
            input_size (int): Dimension of input features.
            hidden_size (int): Dimension of LSTM hidden layers.
            num_layers (int): Number of LSTM layers.
            output_size (int): Length of the output sequence.
            dropout (float): Dropout probability.
        """
        super(AttentionLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        torch.nn.init.xavier_uniform_(self.attention.weight)
        self.attention.bias.data.fill_(0)
        
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, output_size]
        """
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Calculate attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Dropout
        context_vector = self.dropout(context_vector)
        
        # Fully connected layer
        output = self.fc(context_vector)
        
        return output


def create_lstm_model(model_type='basic', input_size=13, hidden_size=64, 
                     num_layers=2, output_size=90, dropout=0.2):
    """
    Create an LSTM model.
    
    Args:
        model_type (str): Model type ('basic', 'bidirectional', 'attention').
        input_size (int): Dimension of input features.
        hidden_size (int): Dimension of hidden layers.
        num_layers (int): Number of LSTM layers.
        output_size (int): Length of the output sequence.
        dropout (float): Dropout probability.
        
    Returns:
        nn.Module: The LSTM model.
    """
    if model_type == 'basic':
        return LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
    elif model_type == 'bidirectional':
        return BidirectionalLSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
    elif model_type == 'attention':
        return AttentionLSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
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
    
    # Test Basic LSTM
    model = create_lstm_model('basic', input_size=input_size, output_size=output_size).to(device)
    output = model(x)
    print(f"Basic LSTM output shape: {output.shape}")
    
    # Test Bidirectional LSTM
    model = create_lstm_model('bidirectional', input_size=input_size, output_size=output_size).to(device)
    output = model(x)
    print(f"Bidirectional LSTM output shape: {output.shape}")
    
    # Test Attention LSTM
    model = create_lstm_model('attention', input_size=input_size, output_size=output_size).to(device)
    output = model(x)
    print(f"Attention LSTM output shape: {output.shape}")

