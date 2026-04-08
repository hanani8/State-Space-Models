
"""
MFCC feature extraction.
"""
import torch
import torch.nn as nn
import torchaudio


class MFCCTransform(nn.Module):
    """
    Extract MFCC features from raw waveform.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        melkwargs: dict = None
    ):
        """
        Args:
            sample_rate: Sample rate of input audio
            n_mfcc: Number of MFCC coefficients
            melkwargs: Arguments for MelSpectrogram
        """
        super().__init__()

        if melkwargs is None:
            melkwargs = {
                'n_fft': 400,        # ~25ms at 16kHz
                'hop_length': 160,    # ~10ms at 16kHz
                'n_mels': 40,
                'f_min': 0,
                'f_max': 8000
            }

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs=melkwargs
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, seq_len) or (seq_len,)

        Returns:
            mfcc: (batch, time, n_mfcc)
        """
        # Handle single sample
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Extract MFCC: (batch, n_mfcc, time)
        mfcc = self.mfcc_transform(waveform)

        # Transpose to (batch, time, n_mfcc)
        mfcc = mfcc.transpose(1, 2)

        return mfcc
