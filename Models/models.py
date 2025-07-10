import torch
import torch.nn as nn

class FlightPricePredictor(nn.Module):
    """
    LSTM-based neural network for predicting flight prices
    
    Args:
        input_dim: Number of input features per time step
        hidden_dim: Size of hidden state in LSTM
        output_dim: Number of output values (usually 1 for price)
    """
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1):
        super(FlightPricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Tensor of shape [batch_size, output_dim] with predicted values
        """
        # x shape: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # Take only the last time step
        last_time_step = lstm_out[:, -1, :]
        # Pass through fully connected layers
        out = self.fc(last_time_step)
        return out
