"""
Utility functions for training and evaluation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
import json
import csv
import os
from collections import defaultdict


def setup_optimizer(model, lr, weight_decay, epochs):
    """
    Setup optimizer with special handling for S4/LSSL parameters.

    S4 parameters can have special _optim attributes that specify
    custom learning rates and weight decay.
    """
    # Collect all parameters
    all_parameters = list(model.parameters())

    # General parameters (no special _optim attribute)
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create optimizer with general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameter groups with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts

    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})

    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    return optimizer, scheduler


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MetricsLogger:
    """Logger for training metrics."""

    def __init__(self, log_dir: str, exp_name: str):
        self.log_dir = log_dir
        self.exp_name = exp_name
        os.makedirs(log_dir, exist_ok=True)

        self.csv_path = os.path.join(log_dir, f"{exp_name}_metrics.csv")
        self.json_path = os.path.join(log_dir, f"{exp_name}_metrics.json")

        self.metrics = defaultdict(list)
        self.csv_file = None
        self.csv_writer = None

    def log(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for an epoch."""
        metrics['epoch'] = epoch

        # Append to metrics dict
        for key, value in metrics.items():
            self.metrics[key].append(value)

        # Write to CSV
        if self.csv_file is None:
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=metrics.keys())
            self.csv_writer.writeheader()

        self.csv_writer.writerow(metrics)
        self.csv_file.flush()

        # Write to JSON
        with open(self.json_path, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)

    def close(self):
        """Close logger."""
        if self.csv_file:
            self.csv_file.close()


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on a dataset.

    Returns:
        loss, accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy
