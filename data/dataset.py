"""
SPEECHCOMMANDS dataset loader with configurable preprocessing.
"""
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from typing import Tuple, Optional


class SpeechCommandsDataset(Dataset):
    """
    Wrapper for SPEECHCOMMANDS dataset with preprocessing.
    """

    def __init__(
        self,
        root: str = './data',
        subset: str = 'training',
        sample_rate: int = 16000,
        max_length: int = 16000,
        download: bool = True
    ):
        """
        Args:
            root: Root directory for dataset
            subset: 'training', 'validation', or 'testing'
            sample_rate: Target sample rate (Hz)
            max_length: Fixed sequence length (samples)
            download: Whether to download dataset
        """
        self.sample_rate = sample_rate
        self.max_length = max_length

        # Load SPEECHCOMMANDS dataset
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=root,
            download=download,
            subset=subset
        )

        # Fixed class labels for SPEECHCOMMANDS v0.02 (35 classes)
        # Much faster than iterating through entire dataset
        self.classes = [
            'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
            'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
            'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila',
            'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
        ]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            waveform: (seq_len,) normalized waveform
            label: integer class label
        """
        waveform, sample_rate, label, *_ = self.dataset[idx]

        # Resample if needed
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        else:
            waveform = waveform

        # Squeeze channel dimension
        waveform = waveform.squeeze(0)

        # Pad or trim to fixed length
        if waveform.shape[0] < self.max_length:
            # Pad with zeros
            waveform = torch.nn.functional.pad(
                waveform,
                (0, self.max_length - waveform.shape[0])
            )
        else:
            # Trim
            waveform = waveform[:self.max_length]

        # Normalize
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()

        # Convert label to index
        label_idx = self.class_to_idx[label]

        return waveform, label_idx


def get_dataloaders(
    root: str = './data',
    batch_size: int = 32,
    num_workers: int = 4,
    sample_rate: int = 16000,
    max_length: int = 16000,
    download: bool = True,
    subset_fraction: float = 1.0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create train, validation, and test dataloaders.

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    # Create datasets
    train_dataset = SpeechCommandsDataset(
        root=root,
        subset='training',
        sample_rate=sample_rate,
        max_length=max_length,
        download=download
    )

    num_classes = train_dataset.num_classes

    if subset_fraction < 1.0:
        n = len(train_dataset)
        k = max(1, int(n * subset_fraction))
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n, generator=rng)[:k].tolist()
        train_dataset = Subset(train_dataset, indices)

    val_dataset = SpeechCommandsDataset(
        root=root,
        subset='validation',
        sample_rate=sample_rate,
        max_length=max_length,
        download=download
    )

    test_dataset = SpeechCommandsDataset(
        root=root,
        subset='testing',
        sample_rate=sample_rate,
        max_length=max_length,
        download=download
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, num_classes
