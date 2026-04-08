# PyTorch Lightning Training Guide

## Overview

The new `train_lightning.py` script provides:
- ✅ **Automatic multi-GPU training** with DDP
- ✅ **Timestamp-based experiment tracking** (no more overwrites!)
- ✅ **Clean checkpointing** with full reproducibility metadata
- ✅ **Better logging** with TensorBoard integration
- ✅ **Mixed precision training** support (16-bit, bfloat16)

## Quick Start

### Single GPU
```bash
conda activate ssm
python train_lightning.py --model lstm --input_type raw
```

### Multi-GPU (Auto-detect all GPUs)
```bash
python train_lightning.py --model lstm --input_type raw
```
Lightning will automatically use all available GPUs!

### Specific number of GPUs
```bash
python train_lightning.py --model lstm --input_type raw --gpus 2
```

### Mixed Precision Training (Faster)
```bash
python train_lightning.py --model lstm --input_type raw --precision 16
```

## Timestamp-based Saving

Each run gets a **unique timestamped directory**:

```
logs/
  lstm_raw_20260407_143022/
    best-epoch=XX-val_acc=XX.XX.ckpt  # Best checkpoint
    last.ckpt                          # Latest checkpoint
    metadata.yaml                      # Full config + CLI args
  lstm_raw_20260407_145533/
    best-epoch=XX-val_acc=XX.XX.ckpt
    last.ckpt
    metadata.yaml
```

**No more overwrites!** Every run is preserved.

## What Gets Saved in Checkpoints

Each checkpoint includes:
- Model weights
- Optimizer state
- Full config (merged with CLI overrides)
- CLI arguments used
- Timestamp
- Number of parameters
- Number of classes

You can reproduce any run exactly!

## Experiment Hyperparameter Overrides

```bash
# Change number of layers
python train_lightning.py --model lstm --input_type raw --epochs 50

# Override learning rate
python train_lightning.py --model transformer --input_type mfcc --learning_rate 0.0005

# Override batch size
python train_lightning.py --model lssl --input_type raw --batch_size 128
```

The merged config is automatically saved to `metadata.yaml`.

## Running All Experiments

```bash
# Multi-GPU version (recommended)
bash run_all_experiments_lightning.sh

# Single GPU version (legacy)
bash run_all_experiments.sh
```

## TensorBoard

View all runs:
```bash
tensorboard --logdir runs
```

Compare different runs in TensorBoard's scalars/hparams tabs!

## Examples

### Compare LSTM architectures
```bash
# 2 layers
python train_lightning.py --model lstm --input_type raw

# 4 layers (different run, both saved!)
python train_lightning.py --model lstm --input_type raw

# Check config in logs/lstm_raw_TIMESTAMP/metadata.yaml to see exact settings
```

### LSSL variants
```bash
python train_lightning.py --model lssl --input_type raw --lssl_variant vanilla
python train_lightning.py --model lssl --input_type raw --lssl_variant hippo_fixed
python train_lightning.py --model lssl --input_type raw --lssl_variant hippo_learned
```

## Multi-GPU Details

Lightning automatically handles:
- DistributedDataParallel (DDP) setup
- Gradient synchronization
- Metric aggregation across GPUs
- Checkpoint saving (only on rank 0)

With 2x L4 GPUs, you'll see:
```
GPU available: True, used: True
TPU available: False
IPU available: False
HPU available: False
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
```

Training will be roughly **2x faster**!

## Old vs New Script

| Feature | `train.py` (old) | `train_lightning.py` (new) |
|---------|------------------|----------------------------|
| Multi-GPU | ❌ No | ✅ Automatic DDP |
| Timestamp saving | ❌ Overwrites | ✅ Unique per run |
| Mixed precision | ❌ No | ✅ 16/32/bf16 |
| Checkpoint metadata | Partial | ✅ Full (config + CLI) |
| Code complexity | More boilerplate | Cleaner |

Both scripts work, but Lightning is recommended for multi-GPU and better experiment tracking.
