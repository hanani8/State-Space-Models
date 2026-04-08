"""
LSTM-based sequence model.
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Strong LSTM implementation with bidirectional layers and dropout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_classes: int = 35,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pooling: str = 'mean'
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            bidirectional: Whether to use bidirectional LSTM
            dropout: Dropout rate
            pooling: 'mean' or 'last' for sequence pooling
        """
        super().__init__()

        self.pooling = pooling
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Classifier
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, features)

        Returns:
            logits: (batch, num_classes)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, time, hidden_dim * num_directions)

        # Pool sequence
        if self.pooling == 'mean':
            # Mean pooling over time
            pooled = torch.mean(lstm_out, dim=1)
        elif self.pooling == 'last':
            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward final states
                h_n = h_n.view(self.lstm.num_layers, 2, x.size(0), self.lstm.hidden_size)
                pooled = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=1)
            else:
                pooled = h_n[-1]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Classify
        logits = self.classifier(pooled)

        return logits
