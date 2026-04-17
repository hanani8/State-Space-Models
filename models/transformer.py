"""
Transformer encoder model for sequence classification.
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 20000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x: (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer encoder for sequence classification.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = 35,
        dropout: float = 0.1,
        pooling: str = 'mean',
        use_positional_encoding: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            num_classes: Number of output classes
            dropout: Dropout rate
            pooling: 'mean' or 'cls' for sequence pooling
            use_positional_encoding: Whether to use positional encoding
        """
        super().__init__()

        self.pooling = pooling
        self.use_positional_encoding = use_positional_encoding

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # CLS token for classification (if using cls pooling)
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, features)

        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.size(0)

        # Project input to model dimension
        x = self.input_projection(x)  # (batch, time, d_model)

        # Add CLS token if using cls pooling
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch, time+1, d_model)

        # Add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, time, d_model)

        # Pool sequence
        if self.pooling == 'mean':
            pooled = torch.mean(x, dim=1)
        elif self.pooling == 'cls':
            pooled = x[:, 0]  # Take CLS token
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Classify
        logits = self.classifier(pooled)

        return logits
