"""
SPEECHCOMMANDS dataset loader with configurable preprocessing and class filtering.
"""
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, Optional, List


class SpeechCommandsDataset(Dataset):
    """
    Wrapper for SPEECHCOMMANDS v0.02 with preprocessing and optional class filtering.

    When `classes` is provided, only those classes are included and labels are
    remapped to consecutive indices [0, len(classes)).  The canonical ordering of
    ALL_CLASSES is preserved so indices are consistent across train/val/test.
    """

    ALL_CLASSES: List[str] = [
        'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
        'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
        'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila',
        'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
    ]

    def __init__(
        self,
        root: str = './data',
        subset: str = 'training',
        sample_rate: int = 16000,
        max_length: int = 16000,
        download: bool = True,
        classes: Optional[List[str]] = None,
    ):
        """
        Args:
            root: Root directory for dataset storage
            subset: 'training', 'validation', or 'testing'
            sample_rate: Target sample rate (Hz)
            max_length: Fixed sequence length in samples (pad/trim)
            download: Download dataset if missing
            classes: Class names to include (None = all 35)
        """
        self.sample_rate = sample_rate
        self.max_length  = max_length

        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=root,
            download=download,
            subset=subset,
        )

        if classes is not None:
            invalid = set(classes) - set(self.ALL_CLASSES)
            if invalid:
                raise ValueError(
                    f"Unknown classes: {sorted(invalid)}\nValid: {self.ALL_CLASSES}"
                )
            # Preserve canonical ordering for consistent label indices across splits
            self.classes = [c for c in self.ALL_CLASSES if c in set(classes)]
        else:
            self.classes = list(self.ALL_CLASSES)

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes   = len(self.classes)

        # Build a filtered index list using file paths — no audio loaded here.
        # _walker entries are absolute paths: .../speech_commands_v0.02/<word>/<file>.wav
        if classes is not None:
            classes_set  = set(self.classes)
            self._indices = [
                i for i, path in enumerate(self.dataset._walker)
                if os.path.basename(os.path.dirname(path)) in classes_set
            ]
        else:
            self._indices = list(range(len(self.dataset)))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            waveform: (max_length,) normalized float32 tensor
            label_idx: integer class index in [0, num_classes)
        """
        waveform, sample_rate, label, *_ = self.dataset[self._indices[idx]]

        if sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.squeeze(0)

        if waveform.shape[0] < self.max_length:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.max_length - waveform.shape[0])
            )
        else:
            waveform = waveform[:self.max_length]

        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()

        return waveform, self.class_to_idx[label]


def get_dataloaders(
    root: str = './data',
    batch_size: int = 32,
    num_workers: int = 4,
    sample_rate: int = 16000,
    max_length: int = 16000,
    download: bool = True,
    subset_fraction: float = 1.0,
    seed: int = 42,
    classes: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create train, validation, and test dataloaders.

    Args:
        classes: Subset of class names (None = all 35).
                 Labels are remapped to [0, len(classes)).

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    shared = dict(
        root=root, sample_rate=sample_rate,
        max_length=max_length, download=download, classes=classes,
    )

    train_dataset = SpeechCommandsDataset(subset='training',   **shared)
    num_classes   = train_dataset.num_classes

    if subset_fraction < 1.0:
        n       = len(train_dataset)
        k       = max(1, int(n * subset_fraction))
        rng     = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n, generator=rng)[:k].tolist()
        train_dataset = Subset(train_dataset, indices)

    val_dataset  = SpeechCommandsDataset(subset='validation', **shared)
    test_dataset = SpeechCommandsDataset(subset='testing',    **shared)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True, batch_size=batch_size)

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, num_classes
