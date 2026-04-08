# Speech Classification Benchmark

A modular PyTorch framework for benchmarking sequence models (LSTM, Transformer, LSSL/S4) on raw speech classification.

## Features

- **9 Model Configurations**:
  - LSTM: raw waveform, 1D conv frontend, MFCC features
  - Transformer: raw waveform, 1D conv frontend, MFCC features
  - LSSL: vanilla (random init), HiPPO-fixed, HiPPO-learned

- **Clean Architecture**:
  - Modular preprocessing and model components
  - Easy to swap datasets
  - Configurable via YAML or command-line

- **Integrated Logging**:
  - TensorBoard for visualization
  - CSV/JSON metrics export

## Quick Start

### 1. Setup Environment

Run the automated setup script (creates venv and installs all dependencies):

```bash
bash setup.sh
```

Or manually:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install tensorboard pyyaml tqdm scikit-learn einops scipy

# Install S4 dependencies
cd s4
pip install -r requirements.txt
cd ..
```

### 2. Test Setup

```bash
# Activate environment
source venv/bin/activate  # Or: source activate_env.sh

# Run setup test
python test_setup.py
```

### 3. Run Training

**Important:** Always activate the virtual environment first:
```bash
source venv/bin/activate
```

#### LSTM Examples

```bash
# LSTM on raw waveform
python train.py --model lstm --input_type raw

# LSTM with Conv1D frontend
python train.py --model lstm --input_type conv

# LSTM with MFCC features
python train.py --model lstm --input_type mfcc
```

#### Transformer Examples

```bash
# Transformer on raw waveform
python train.py --model transformer --input_type raw

# Transformer with Conv1D frontend
python train.py --model transformer --input_type conv

# Transformer with MFCC features
python train.py --model transformer --input_type mfcc
```

#### LSSL/S4 Examples

```bash
# LSSL with vanilla (random diagonal) initialization
python train.py --model lssl --input_type raw --lssl_variant vanilla

# LSSL with HiPPO initialization (A matrix frozen)
python train.py --model lssl --input_type raw --lssl_variant hippo_fixed

# LSSL with HiPPO initialization (A matrix trainable)
python train.py --model lssl --input_type raw --lssl_variant hippo_learned
```

### 4. Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir runs
```

## Project Structure

```
project/
├── data/
│   └── dataset.py          # SPEECHCOMMANDS dataset loader
├── features/
│   └── mfcc.py             # MFCC feature extraction
├── models/
│   ├── lstm.py             # LSTM model
│   ├── transformer.py      # Transformer encoder
│   ├── lssl.py             # LSSL wrapper (uses S4 repo)
│   └── conv_frontend.py    # 1D Conv frontend
├── s4/                     # Official S4 repository (cloned)
├── config.yaml             # Default configuration
├── train.py                # Main training script
├── utils.py                # Training utilities
└── requirements.txt        # Dependencies
```

## Configuration

Edit `config.yaml` or pass command-line arguments:

```bash
python train.py \
  --model transformer \
  --input_type conv \
  --batch_size 64 \
  --learning_rate 0.001 \
  --epochs 50
```

## Model Details

### LSTM
- Multi-layer bidirectional LSTM
- Configurable hidden dimension and layers
- Mean or last-state pooling

### Transformer
- Multi-head self-attention encoder
- Sinusoidal or learned positional encoding
- Mean or CLS token pooling

### LSSL (Linear State-Space Layer)
Uses the original **LSSL** implementation (NeurIPS 2021) from https://github.com/state-spaces/s4

**Three variants:**

1. **Vanilla**: Random initialization, trainable A matrix
2. **HiPPO-Fixed**: HiPPO (LegS) initialization, **frozen** A matrix
3. **HiPPO-Learned**: HiPPO (LegS) initialization, **trainable** A matrix

**What's LSSL?** The original Linear State-Space Layer (2021), which came **before** S4 and S4D.

**What about S4 and S4D?** See [LSSL_VS_S4.md](LSSL_VS_S4.md) for:
- Timeline and relationships
- Why we use LSSL as the baseline
- How to add S4/S4D as separate models later for comparison

The HiPPO initialization provides theoretically-motivated structured matrices using orthogonal polynomials (Legendre, Laguerre, etc.).

## Output

Each run generates:
- **Checkpoints**: Best model saved in `logs/`
- **Metrics**: CSV and JSON in `logs/`
- **TensorBoard**: Training curves in `runs/`

Metrics logged:
- Training loss and accuracy
- Validation loss and accuracy
- Test accuracy (final)
- Training time

## Citation

This project uses the S4 (Structured State Space) implementation:

```
@inproceedings{gu2022efficiently,
  title={Efficiently Modeling Long Sequences with Structured State Spaces},
  author={Gu, Albert and Goel, Karan and R{\'e}, Christopher},
  booktitle={The International Conference on Learning Representations (ICLR)},
  year={2022}
}
```
