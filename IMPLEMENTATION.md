# Implementation Overview

This document provides a detailed overview of the speech classification benchmark implementation.

## Architecture

### Clean Separation of Concerns

```
Input Pipeline → Preprocessing → Model → Classifier
```

1. **Input Pipeline** (`data/dataset.py`)
   - SPEECHCOMMANDS dataset wrapper
   - Automatic resampling to 16kHz
   - Padding/trimming to fixed length (16000 samples = 1 second)
   - Normalization

2. **Preprocessing** (Optional)
   - **Raw**: No preprocessing, direct waveform
   - **Conv1D Frontend** (`models/conv_frontend.py`): Learned feature extraction
   - **MFCC** (`features/mfcc.py`): Hand-crafted acoustic features

3. **Model** (Sequence encoder)
   - **LSTM** (`models/lstm.py`): Bidirectional, multi-layer RNN
   - **Transformer** (`models/transformer.py`): Self-attention encoder
   - **LSSL** (`models/lssl.py`): State-space model (wraps S4)

4. **Classifier**
   - Sequence pooling (mean or last/CLS)
   - Linear layers with dropout

## Key Design Decisions

### 1. Modular Preprocessing

All preprocessing modules output `(batch, time, features)`:
- Raw: `(B, 16000)` → `(B, 16000, 1)`
- Conv: `(B, 16000)` → `(B, ~2000, 64)` (depends on stride)
- MFCC: `(B, 16000)` → `(B, ~100, 40)`

This consistency allows any model to work with any input type.

### 2. LSSL Integration Strategy

**We DO NOT reimplement S4.** Instead, we:

1. Clone the official S4 repo
2. Add it to Python path
3. Import and wrap their modules:
   - `S4D`: Diagonal SSM (vanilla variant)
   - `S4Block`: Full DPLR SSM (HiPPO variants)

**Three LSSL Variants:**

```python
# Vanilla: Random diagonal initialization
S4D(d_model, d_state, lr=0.001)

# HiPPO Fixed: Structured init, freeze A
S4Block(d_model, d_state, mode='dplr', init='hippo', 
        lr={'A': 0.0, 'B': 0.001, 'dt': 0.001})

# HiPPO Learned: Structured init, train A
S4Block(d_model, d_state, mode='dplr', init='hippo',
        lr={'A': 0.001, 'B': 0.001, 'dt': 0.001})
```

### 3. Optimizer Setup for S4

S4 parameters use special `_optim` attributes:

```python
# Example parameter with custom LR
param._optim = {"lr": 0.001, "weight_decay": 0.0}
```

Our `setup_optimizer()` in `utils.py` handles this:
- Collects parameters with `_optim` attributes
- Creates separate param groups with custom hyperparameters
- Ensures SSM parameters get appropriate learning rates

### 4. Consistent Training Interface

All models follow the same interface:

```python
model = SomeModel(input_dim, num_classes, ...)
logits = model(x)  # x: (batch, time, features)
```

This makes `train.py` model-agnostic.

## Model Details

### LSTM

**Architecture:**
```
Input → LSTM Layers (bidirectional, residual) → Pooling → Classifier
```

**Strengths:**
- Strong baseline for sequential data
- Efficient on CPU
- Well-understood hyperparameters

**Config:**
- `hidden_dim`: 128 (default)
- `num_layers`: 3
- `bidirectional`: True
- Pooling: mean over time

### Transformer

**Architecture:**
```
Input → Projection → Positional Encoding → 
  Transformer Encoder Layers → Pooling → Classifier
```

**Strengths:**
- Parallel processing (fast on GPU)
- Captures long-range dependencies via attention
- Flexible receptive field

**Config:**
- `d_model`: 128
- `nhead`: 8
- `num_layers`: 4
- `dim_feedforward`: 512
- Positional encoding: Sinusoidal

### LSSL (State-Space Models)

**What is LSSL?**
- Original Linear State-Space Layer from NeurIPS 2021
- Foundation for later S4 and S4D models
- Uses Krylov methods for convolution computation

**Architecture:**
```
Input → Projection → LSSL Layers (residual) → Pooling → Classifier

LSSL Layer:
  x → SSM Convolution (via Krylov) → Nonlinearity → Residual
```

**Key Concepts:**

1. **State-Space Equation:**
   ```
   Continuous: x'(t) = Ax(t) + Bu(t), y(t) = Cx(t) + Du(t)
   Discrete:   x_k = A̅x_{k-1} + B̅u_k,  y_k = Cx_k + Du_k
   ```

2. **Discretization:**
   Continuous → Discrete via generalized bilinear transform (GBT)
   ```
   A̅ = (I - Δt/2 · A)^(-1) (I + Δt/2 · A)
   B̅ = (I - Δt/2 · A)^(-1) Δt · B
   ```

3. **Convolution Representation:**
   Compute via Krylov matrix K(A, B) = [B, AB, A²B, ...]
   Then y = conv(K, u)

4. **HiPPO (High-order Polynomial Projection Operators):**
   - Theoretically-motivated initialization for A
   - Encodes history using orthogonal polynomials
   - **LegS** (Legendre Scaled): Bounded memory, most common
   - **LegT** (Legendre Translated): Bounded memory, alternative
   - **LagT** (Laguerre Translated): Unbounded memory

**Variants:**

| Variant | Measure | Learn | A Matrix | Notes |
|---------|---------|-------|----------|-------|
| Vanilla | `'random'` | 1 | Random, trainable | Baseline |
| HiPPO Fixed | `'legs'` | 0 | LegS HiPPO, **frozen** | Structure alone |
| HiPPO Learned | `'legs'` | 1 | LegS HiPPO, trainable | Structure + learning |

**Parameters:**
- `measure`: Initialization method ('random', 'legs', 'legt', 'lagt')
- `learn`: 0 (frozen), 1 (shared A), 2 (separate A per feature)
- `lr`: Learning rate for transition matrices (when trainable)
- `d_model` (N): State dimension (order of HiPPO projection)

**Config:**
- `d_model`: 128 (model/hidden dimension)
- `d_state`: 64 (N, state size for HiPPO)
- `num_layers`: 4
- Special LR for state matrices: 0.001

## Experiment Tracking

### TensorBoard

All metrics logged to `runs/`:
```bash
tensorboard --logdir runs
```

Metrics per epoch:
- Training loss & accuracy
- Validation loss & accuracy
- Learning rate schedule

### CSV/JSON Logs

Saved to `logs/`:
- `{exp_name}_metrics.csv`: Easy spreadsheet analysis
- `{exp_name}_metrics.json`: Programmatic access

### Checkpoints

Best model saved as: `logs/{exp_name}_best.pth`

Contains:
- Model state dict
- Optimizer state
- Best validation accuracy
- Full config

## Running Experiments

### Single Experiment

```bash
python train.py \
  --model lssl \
  --input_type raw \
  --lssl_variant hippo_learned \
  --batch_size 64 \
  --learning_rate 0.001 \
  --epochs 50
```

### All 9 Configurations

```bash
bash run_all_experiments.sh
```

This runs:
1. LSTM + raw
2. LSTM + conv
3. LSTM + mfcc
4. Transformer + raw
5. Transformer + conv
6. Transformer + mfcc
7. LSSL + vanilla
8. LSSL + hippo_fixed
9. LSSL + hippo_learned

## Expected Results

### Performance Characteristics

**LSTM:**
- ✓ Solid baseline
- ✓ Fast inference
- ✗ Sequential (slow training)
- ✗ Gradient issues on very long sequences

**Transformer:**
- ✓ Fast training (parallel)
- ✓ Strong performance
- ✗ Quadratic complexity in sequence length
- ✗ Many parameters

**LSSL/S4:**
- ✓ Linear complexity
- ✓ Excellent long-range modeling
- ✓ Efficient convolution mode
- ✗ More complex implementation
- ? HiPPO init should help on long sequences

### Input Type Trade-offs

**Raw Waveform:**
- ✓ No information loss
- ✓ End-to-end learning
- ✗ Very long sequences (16000 timesteps)
- ✗ Needs more capacity

**Conv Frontend:**
- ✓ Learned features
- ✓ Shorter sequences (~2000 timesteps)
- ✓ Good balance

**MFCC:**
- ✓ Shortest sequences (~100 timesteps)
- ✓ Domain knowledge
- ✗ Hand-crafted (limits learning)
- ✗ Information loss

## Extending the Framework

### Adding a New Model

1. Create `models/your_model.py`:
```python
class YourModel(nn.Module):
    def __init__(self, input_dim, num_classes, ...):
        ...
    
    def forward(self, x):
        # x: (batch, time, features)
        # return: (batch, num_classes)
        ...
```

2. Add to `train.py`:
```python
elif model_type == 'your_model':
    from models.your_model import YourModel
    model = YourModel(input_dim, num_classes, ...)
```

3. Add config to `config.yaml`:
```yaml
your_model:
  param1: value1
  param2: value2
```

### Adding a New Dataset

1. Edit `data/dataset.py`:
```python
def get_dataloaders_custom(root, ...):
    # Load your dataset
    # Return train_loader, val_loader, test_loader, num_classes
    ...
```

2. Ensure output is `(waveform, label)` where:
   - `waveform`: (seq_len,) float tensor
   - `label`: int

## Citation

If you use this codebase or the LSSL models, please cite:

```bibtex
@inproceedings{gu2022efficiently,
  title={Efficiently Modeling Long Sequences with Structured State Spaces},
  author={Gu, Albert and Goel, Karan and R{\'e}, Christopher},
  booktitle={ICLR},
  year={2022}
}
```
