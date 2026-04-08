# Quick Setup Guide for New GPU

Follow these steps to get started on a fresh GPU instance:

## 1. Clone Repository

```bash
git clone https://github.com/hanani8/State-Space-Models.git
cd State-Space-Models

# Clone S4 repository (required for LSSL model)
git clone https://github.com/state-spaces/s4.git

# Create required __init__.py files in S4
touch s4/src/models/__init__.py
touch s4/src/models/sequence/modules/__init__.py
```

## 2. Setup Conda Environment

### Option A: Use existing conda environment (if available)

If you have a conda environment called "ssm":
```bash
conda activate ssm
```

### Option B: Create new conda environment

```bash
# Create environment with Python 3.14
conda create -n ssm python=3.14 -y
conda activate ssm

# Install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install -r requirements.txt
pip install pytorch-lightning tensorboard
```

## 3. Verify Setup

```bash
python test_setup.py
```

You should see:
```
✅ All tests passed! Setup is complete.
```

## 4. Download Dataset (First Run Only)

The Speech Commands dataset (~2.3GB) will auto-download on first training run.

To pre-download:
```bash
python -c "from data.dataset import get_dataloaders; get_dataloaders()"
```

This takes ~5-10 minutes depending on internet speed.

## 5. Start Training

### Single experiment
```bash
# Single GPU
python train_lightning.py --model lstm --input_type raw

# Multi-GPU (auto-detect)
python train_lightning.py --model lstm --input_type raw

# Specific number of GPUs
python train_lightning.py --model lstm --input_type raw --gpus 2
```

### All 9 experiments
```bash
bash run_all_experiments_lightning.sh
```

## 6. Monitor Training

In another terminal:
```bash
tensorboard --logdir runs
```

Then open http://localhost:6006 in your browser.

## GPU Requirements

**Minimum:**
- 1x GPU with 8GB VRAM
- CUDA 11.8+

**Recommended:**
- 1x L4 (24GB) or T4 (16GB)
- 2x GPUs for faster training

**Estimated time for all 9 experiments:**
- 1x L4: ~8 hours
- 2x L4: ~4 hours
- 1x T4: ~12 hours

## Troubleshooting

### S4 submodule not cloned
```bash
git submodule update --init --recursive
```

### CUDA out of memory
Reduce batch size in `config.yaml`:
```yaml
data:
  batch_size: 32  # Try 16 or 8
```

### Dataset download issues
Manually download from: https://pytorch.org/audio/stable/datasets.html#speechcommands

### Import errors
Make sure you're in the conda environment:
```bash
conda activate ssm
which python  # Should point to conda env
```

## File Structure

```
State-Space-Models/
├── config.yaml              # Main configuration
├── train_lightning.py       # PyTorch Lightning training (multi-GPU)
├── train.py                 # Legacy training (single GPU)
├── test_setup.py           # Verify installation
├── models/                  # Model implementations
│   ├── lstm.py
│   ├── transformer.py
│   └── lssl.py             # State-space model
├── data/                    # Dataset loaders
├── features/                # Feature extraction
├── s4/                      # S4 implementation (submodule)
└── logs/                    # Training outputs (created on first run)
```

## Cost Estimate

For all 9 experiments on cloud GPUs:
- Vast.ai (1x L4): ~₹500-700 (~$6-8)
- RunPod (1x L4): ~₹800-1000 (~$10-12)

## Next Steps

See `LIGHTNING_GUIDE.md` for detailed training options.
