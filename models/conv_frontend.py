"""
1D Convolutional frontend for learned feature extraction.
"""
import torch
import torch.nn as nn


class Conv1DFrontend(nn.Module):
    """
    Stack of 1D convolutions as a learned feature extractor.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: list = [32, 64],
        kernel_size: int = 5,
        stride: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            in_channels: Number of input channels (1 for raw audio)
            hidden_channels: List of channel sizes for each conv layer
            kernel_size: Kernel size for convolutions
            stride: Stride for convolutions
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        prev_channels = in_channels

        for i, out_channels in enumerate(hidden_channels):
            layers.extend([
                nn.Conv1d(
                    prev_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_channels = out_channels

        self.conv_stack = nn.Sequential(*layers)
        self.out_channels = hidden_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) raw waveform

        Returns:
            features: (batch, time, features)
        """
        # Add channel dimension: (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Apply convolutions: (batch, channels, time)
        x = self.conv_stack(x)

        # Transpose to (batch, time, features)
        x = x.transpose(1, 2)

        return x
