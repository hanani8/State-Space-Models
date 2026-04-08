"""
PyTorch Lightning training script for speech classification.
Supports multi-GPU training with automatic DDP and timestamp-based saving.

Usage:
    # Single GPU
    python train_lightning.py --model lstm --input_type raw

    # Multi-GPU
    python train_lightning.py --model lstm --input_type raw --gpus 2
"""
import argparse
import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import os

from data.dataset import get_dataloaders
from models.lstm import LSTMModel
from models.transformer import TransformerModel
from models.lssl import LSSLModel
from models.conv_frontend import Conv1DFrontend
from features.mfcc import MFCCTransform
from utils import count_parameters


class SpeechClassificationModule(pl.LightningModule):
    """PyTorch Lightning module for speech classification."""

    def __init__(self, config, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.num_classes = num_classes

        # Build model
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()

    def _build_model(self):
        """Build model based on configuration."""
        config = self.config
        model_type = config['model']
        input_type = config['input_type']

        # Determine input dimension
        if input_type == 'raw':
            input_dim = 1
        elif input_type == 'conv':
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

        # Build backbone model
        if model_type == 'lstm':
            model_config = config['lstm']
            backbone = LSTMModel(
                input_dim=input_dim,
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                num_classes=self.num_classes,
                bidirectional=model_config['bidirectional'],
                dropout=model_config['dropout'],
                pooling=model_config['pooling']
            )
        elif model_type == 'transformer':
            model_config = config['transformer']
            backbone = TransformerModel(
                input_dim=input_dim,
                d_model=model_config['d_model'],
                nhead=model_config['nhead'],
                num_layers=model_config['num_layers'],
                dim_feedforward=model_config['dim_feedforward'],
                num_classes=self.num_classes,
                dropout=model_config['dropout'],
                pooling=model_config['pooling']
            )
        elif model_type == 'lssl':
            model_config = config['lssl']
            backbone = LSSLModel(
                input_dim=input_dim,
                d_model=model_config['d_model'],
                d_state=model_config['d_state'],
                num_layers=model_config['num_layers'],
                num_classes=self.num_classes,
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
                    if self.input_type in ['conv', 'mfcc']:
                        x = self.preprocessor(x)
                    else:
                        x = x.unsqueeze(-1)
                    return self.model(x)

            return ModelWithPreprocessing(preprocessor, backbone, input_type)
        else:
            class RawInputModel(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    x = x.unsqueeze(-1)
                    return self.model(x)

            return RawInputModel(backbone)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        # Calculate accuracy
        _, predicted = outputs.max(1)
        acc = (predicted == targets).float().mean()

        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc * 100, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        # Calculate accuracy
        _, predicted = outputs.max(1)
        acc = (predicted == targets).float().mean()

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc * 100, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        # Calculate accuracy
        _, predicted = outputs.max(1)
        acc = (predicted == targets).float().mean()

        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc * 100, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['epochs']
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


class SpeechDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for speech classification."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.num_classes = None

    def setup(self, stage=None):
        """Load data. Called on every GPU in distributed training."""
        self.train_loader, self.val_loader, self.test_loader, self.num_classes = get_dataloaders(
            root=self.config['data']['root'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            sample_rate=self.config['data']['sample_rate'],
            max_length=self.config['data']['max_length']
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


def main():
    parser = argparse.ArgumentParser(description='Train speech classification models with PyTorch Lightning')
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
    parser.add_argument('--gpus', type=int, default=None,
                        help='Number of GPUs (0 for CPU, -1 for all available)')
    parser.add_argument('--precision', type=str, default='32',
                        help='Training precision: 32, 16, bf16')

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

    # Set seed
    pl.seed_everything(config['training']['seed'])

    # Create experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config['model']}_{config['input_type']}"
    if config['model'] == 'lssl':
        exp_name += f"_{config['lssl']['lssl_variant']}"
    exp_name_with_time = f"{exp_name}_{timestamp}"

    print(f"\n{'='*80}")
    print(f"Experiment: {exp_name_with_time}")
    print(f"Configuration:")
    print(f"  Model: {config['model']}")
    print(f"  Input: {config['input_type']}")
    if config['model'] == 'lssl':
        print(f"  LSSL Variant: {config['lssl']['lssl_variant']}")
    print(f"  GPUs: {args.gpus if args.gpus is not None else 'auto'}")
    print(f"  Precision: {args.precision}")
    print(f"{'='*80}\n")

    # Setup data module
    print("Setting up data module...")
    data_module = SpeechDataModule(config)
    data_module.setup()

    print(f"Number of classes: {data_module.num_classes}")
    print(f"Train batches: {len(data_module.train_loader)}")
    print(f"Val batches: {len(data_module.val_loader)}")
    print(f"Test batches: {len(data_module.test_loader)}\n")

    # Build model
    print("Building model...")
    model = SpeechClassificationModule(config, data_module.num_classes)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}\n")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['logging']['log_dir'], exp_name_with_time),
        filename='best-{epoch:02d}-{val_acc:.2f}',
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=config['logging']['tensorboard_dir'],
        name=exp_name,
        version=timestamp,
    )

    # Determine GPU configuration
    if args.gpus is not None:
        devices = args.gpus
    else:
        # Auto-detect: use all GPUs if available, otherwise CPU
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator='gpu' if devices > 0 else 'cpu',
        devices=devices if devices > 0 else 1,
        strategy='ddp' if devices > 1 else 'auto',
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=config['logging']['print_freq'],
        deterministic=True,
    )

    # Train
    print(f"Starting training for {config['training']['epochs']} epochs...")
    print(f"Using {'CPU' if devices == 0 else f'{devices} GPU(s)'}\n")

    trainer.fit(model, data_module)

    # Test with best checkpoint
    print("\nEvaluating on test set...")
    trainer.test(model, data_module, ckpt_path='best')

    # Save final metadata
    metadata_path = os.path.join(config['logging']['log_dir'], exp_name_with_time, 'metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.dump({
            'timestamp': timestamp,
            'experiment_name': exp_name_with_time,
            'config': config,
            'cli_args': vars(args),
            'num_parameters': num_params,
            'num_classes': data_module.num_classes,
        }, f, default_flow_style=False)

    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Checkpoints saved to: {os.path.join(config['logging']['log_dir'], exp_name_with_time)}")
    print(f"TensorBoard logs: {os.path.join(config['logging']['tensorboard_dir'], exp_name, timestamp)}")
    print(f"View logs with: tensorboard --logdir {config['logging']['tensorboard_dir']}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
