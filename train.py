"""
Main training script for speech classification benchmarking.
"""
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import os
import random
import numpy as np

from data.dataset import get_dataloaders
from models.lstm import LSTMModel
from models.transformer import TransformerModel
from models.lssl import LSSLModel
from models.conv_frontend import Conv1DFrontend
from features.mfcc import MFCCTransform
from utils import setup_optimizer, count_parameters, MetricsLogger, evaluate


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(config, num_classes, device):
    """Build model based on configuration."""
    model_type = config['model']
    input_type = config['input_type']

    # Determine input dimension
    if input_type == 'raw':
        input_dim = 1  # Raw waveform needs projection
    elif input_type == 'conv':
        # Conv frontend output
        conv_config = config['conv_frontend']
        input_dim = conv_config['hidden_channels'][-1]
    elif input_type == 'mfcc':
        input_dim = config['mfcc']['n_mfcc']
    else:
        raise ValueError(f"Unknown input type: {input_type}")

    # Build preprocessing
    if input_type == 'conv':
        preprocessor = Conv1DFrontend(
            in_channels=1,
            hidden_channels=conv_config['hidden_channels'],
            kernel_size=conv_config['kernel_size'],
            stride=conv_config['stride'],
            dropout=0.1
        )
    elif input_type == 'mfcc':
        preprocessor = MFCCTransform(
            sample_rate=config['data']['sample_rate'],
            n_mfcc=config['mfcc']['n_mfcc']
        )
    else:
        preprocessor = None

    # Build model
    if model_type == 'lstm':
        model_config = config['lstm']
        model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_classes=num_classes,
            bidirectional=model_config['bidirectional'],
            dropout=model_config['dropout'],
            pooling=model_config['pooling']
        )
    elif model_type == 'transformer':
        model_config = config['transformer']
        model = TransformerModel(
            input_dim=input_dim,
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_layers=model_config['num_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            num_classes=num_classes,
            dropout=model_config['dropout'],
            pooling=model_config['pooling']
        )
    elif model_type == 'lssl':
        model_config = config['lssl']
        model = LSSLModel(
            input_dim=input_dim,
            d_model=model_config['d_model'],
            d_state=model_config['d_state'],
            num_layers=model_config['num_layers'],
            num_classes=num_classes,
            dropout=model_config['dropout'],
            pooling=model_config['pooling'],
            lssl_variant=model_config['lssl_variant'],
            prenorm=model_config['prenorm']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Combine preprocessor and model
    if preprocessor is not None:
        class ModelWithPreprocessing(nn.Module):
            def __init__(self, preprocessor, model, input_type):
                super().__init__()
                self.preprocessor = preprocessor
                self.model = model
                self.input_type = input_type

            def forward(self, x):
                # x: (batch, seq_len) for raw audio
                if self.input_type in ['conv', 'mfcc']:
                    x = self.preprocessor(x)  # (batch, time, features)
                else:
                    # For raw: add feature dimension
                    x = x.unsqueeze(-1)  # (batch, seq_len, 1)
                return self.model(x)

        full_model = ModelWithPreprocessing(preprocessor, model, input_type)
    else:
        # Raw input without conv frontend
        class RawInputModel(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                # x: (batch, seq_len)
                x = x.unsqueeze(-1)  # (batch, seq_len, 1)
                return self.model(x)

        full_model = RawInputModel(model)

    full_model = full_model.to(device)
    return full_model


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, print_freq=10):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        if (batch_idx + 1) % print_freq == 0:
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{acc:.2f}%'
            })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train speech classification models')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                        help='Model type: lstm, transformer, lssl')
    parser.add_argument('--input_type', type=str, default=None,
                        help='Input type: raw, conv, mfcc')
    parser.add_argument('--lssl_variant', type=str, default=None,
                        help='LSSL variant: vanilla, hippo_fixed, hippo_learned')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cuda or cpu')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command-line arguments
    if args.model:
        config['model'] = args.model
    if args.input_type:
        config['input_type'] = args.input_type
    if args.lssl_variant:
        config['lssl']['lssl_variant'] = args.lssl_variant
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.seed:
        config['training']['seed'] = args.seed
    if args.device:
        config['device'] = args.device

    # Set seed
    set_seed(config['training']['seed'])

    # Setup device
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"\n{'='*80}")
    print(f"Configuration:")
    print(f"  Model: {config['model']}")
    print(f"  Input: {config['input_type']}")
    if config['model'] == 'lssl':
        print(f"  LSSL Variant: {config['lssl']['lssl_variant']}")
    print(f"  Device: {device}")
    print(f"{'='*80}\n")

    # Get dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        root=config['data']['root'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        sample_rate=config['data']['sample_rate'],
        max_length=config['data']['max_length']
    )
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}\n")

    # Build model
    print("Building model...")
    model = build_model(config, num_classes, device)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}\n")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = setup_optimizer(
        model,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        epochs=config['training']['epochs']
    )

    # Setup logging
    exp_name = f"{config['model']}_{config['input_type']}"
    if config['model'] == 'lssl':
        exp_name += f"_{config['lssl']['lssl_variant']}"

    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['tensorboard_dir'], exist_ok=True)

    logger = MetricsLogger(config['logging']['log_dir'], exp_name)
    writer = SummaryWriter(os.path.join(config['logging']['tensorboard_dir'], exp_name))

    # Training loop
    print(f"Starting training for {config['training']['epochs']} epochs...\n")
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, config['training']['epochs'] + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, config['logging']['print_freq']
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log metrics
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        }
        logger.log(epoch, metrics)

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Print summary
        print(f"Epoch {epoch}/{config['training']['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(
                config['logging']['log_dir'],
                f"{exp_name}_best.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, checkpoint_path)
            print(f"  → Saved best model (val_acc: {val_acc:.2f}%)")

        print()

    # Final test evaluation
    print("Evaluating on test set...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*80}")
    print(f"Final Results:")
    print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Training Time: {(time.time() - start_time) / 60:.2f} minutes")
    print(f"{'='*80}\n")

    # Log final results
    writer.add_hparams(
        {
            'model': config['model'],
            'input_type': config['input_type'],
            'lr': config['training']['learning_rate'],
            'batch_size': config['data']['batch_size'],
        },
        {
            'hparam/val_acc': best_val_acc,
            'hparam/test_acc': test_acc,
        }
    )

    # Cleanup
    logger.close()
    writer.close()

    print(f"Logs saved to: {config['logging']['log_dir']}")
    print(f"TensorBoard logs: {config['logging']['tensorboard_dir']}")


if __name__ == '__main__':
    main()
