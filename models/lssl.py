"""
LSSL (Linear State-Space Layer) wrapper using the official LSSL implementation.

LSSL is from the NeurIPS 2021 paper:
"Combining Recurrent, Convolutional, and Continuous-time Models with Linear State Space Layers"
by Gu et al.

This is the ORIGINAL state-space layer, which came BEFORE S4 and S4D.

Three variants:
1. Vanilla: Random initialization, trainable
2. HiPPO Fixed: HiPPO initialization (structured), NOT trainable
3. HiPPO Learned: HiPPO initialization (structured), trainable
"""
import sys
import os
import torch
import torch.nn as nn

# Add S4 repo root to path (so 'import src' works like in their train.py)
S4_PATH = os.path.join(os.path.dirname(__file__), '..', 's4')
S4_PATH = os.path.abspath(S4_PATH)
if S4_PATH not in sys.path:
    sys.path.insert(0, S4_PATH)

# Import LSSL from the S4 implementation (using 'src' prefix like their train.py)
from src.models.sequence.modules.lssl import LSSL


class LSSLModel(nn.Module):
    """
    LSSL-based sequence classification model.

    Uses the original LSSL implementation (NeurIPS 2021).
    Later you can compare against S4 and S4D separately.

    Supports three initialization variants:
    - 'vanilla': Random initialization, trainable
    - 'hippo_fixed': HiPPO (LegS) initialization, frozen
    - 'hippo_learned': HiPPO (LegS) initialization, trainable
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        d_state: int = 64,
        num_layers: int = 4,
        num_classes: int = 35,
        dropout: float = 0.1,
        pooling: str = 'mean',
        lssl_variant: str = 'vanilla',
        prenorm: bool = False
    ):
        """
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension (H in LSSL paper)
            d_state: State dimension (N in LSSL paper - order of HiPPO)
            num_layers: Number of LSSL layers
            num_classes: Number of output classes
            dropout: Dropout rate
            pooling: 'mean' for mean pooling over time
            lssl_variant: 'vanilla', 'hippo_fixed', or 'hippo_learned'
            prenorm: Use prenorm instead of postnorm
        """
        super().__init__()

        self.prenorm = prenorm
        self.pooling = pooling
        self.lssl_variant = lssl_variant

        # Input projection
        self.encoder = nn.Linear(input_dim, d_model)

        # Build LSSL layers based on variant
        self.lssl_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(num_layers):
            if lssl_variant == 'vanilla':
                # Vanilla: Random initialization, trainable
                layer = LSSL(
                    d=d_model,              # Model dimension
                    d_model=d_state,        # State dimension (N)
                    measure='random',       # Random initialization
                    learn=1,                # Trainable (1 = shared A, 2 = separate A per feature)
                    lr=0.001,              # Learning rate for state matrix
                    channels=1,            # Output channels
                    activation='gelu',
                    dropout=dropout,
                )
            elif lssl_variant == 'hippo_fixed':
                # HiPPO Fixed: Structured initialization, NOT trainable
                layer = LSSL(
                    d=d_model,
                    d_model=d_state,
                    measure='legs',        # LegS = Legendre (scaled) - HiPPO method
                    learn=0,               # NOT trainable (frozen)
                    channels=1,
                    activation='gelu',
                    dropout=dropout,
                )
            elif lssl_variant == 'hippo_learned':
                # HiPPO Learned: Structured initialization, trainable
                layer = LSSL(
                    d=d_model,
                    d_model=d_state,
                    measure='legs',        # LegS = Legendre (scaled) - HiPPO method
                    learn=1,               # Trainable
                    lr=0.001,              # Learning rate for state matrix
                    channels=1,
                    activation='gelu',
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unknown LSSL variant: {lssl_variant}")

            self.lssl_layers.append(layer)
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, features)

        Returns:
            logits: (batch, num_classes)
        """
        # Project input
        x = self.encoder(x)  # (B, L, d_model)

        # Apply LSSL layers with residual connections
        # Note: LSSL expects (B, L, H) format (matches our input)
        for layer, norm, dropout in zip(self.lssl_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z)

            # Apply LSSL layer (returns output and state, we ignore state)
            z, _ = layer(z)

            # Dropout
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                x = norm(x)

        # Pool sequence
        if self.pooling == 'mean':
            pooled = torch.mean(x, dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Classify
        logits = self.classifier(pooled)

        return logits
