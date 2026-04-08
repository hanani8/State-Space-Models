# Quick Reference

## What Models Are Implemented?

### Currently Available (9 configurations)

1. **LSTM** + Raw waveform
2. **LSTM** + Conv1D frontend
3. **LSTM** + MFCC features
4. **Transformer** + Raw waveform
5. **Transformer** + Conv1D frontend
6. **Transformer** + MFCC features
7. **LSSL** (vanilla: random init, trainable)
8. **LSSL** (HiPPO-fixed: LegS init, frozen)
9. **LSSL** (HiPPO-learned: LegS init, trainable)

### Future Extensions

10. **S4** (ICLR 2022) - Add as separate model
11. **S4D** (NeurIPS 2022) - Add as separate model

---

## What is LSSL?

**LSSL** = Linear State-Space Layer (NeurIPS 2021)

- The **original** trainable state-space model
- Came **before** S4 and S4D
- Uses full N×N state matrix (not restricted to diagonal)
- Foundation for all modern state-space models

**Not the same as S4 or S4D!**

---

## LSSL vs S4 vs S4D

| Model | Year | A Matrix | Speed | Parameters |
|-------|------|----------|-------|------------|
| **LSSL** | 2021 | Full N×N | Slower | More |
| **S4** | 2022 | Diagonal + Low-Rank (DPLR) | Fast | Medium |
| **S4D** | 2022 | Diagonal only | Fastest | Fewer |

**See [LSSL_VS_S4.md](LSSL_VS_S4.md) for detailed comparison.**

---

## Quick Start Commands

### Setup
```bash
bash setup.sh
source venv/bin/activate
python test_setup.py
```

### Single Experiment
```bash
# LSTM
python train.py --model lstm --input_type raw

# Transformer  
python train.py --model transformer --input_type mfcc

# LSSL (vanilla)
python train.py --model lssl --input_type raw --lssl_variant vanilla

# LSSL (HiPPO, learned)
python train.py --model lssl --input_type raw --lssl_variant hippo_learned
```

### All Experiments
```bash
bash run_all_experiments.sh
```

### View Results
```bash
tensorboard --logdir runs
```

---

## LSSL Variants Explained

### 1. Vanilla
```python
LSSL(measure='random', learn=1)
```
- Random initialization
- Trainable A matrix
- **Use case**: Baseline to test if structure helps

### 2. HiPPO-Fixed
```python
LSSL(measure='legs', learn=0)
```
- HiPPO (LegS) initialization
- **Frozen** A matrix (not trainable)
- **Use case**: Test if structure alone (without learning) helps

### 3. HiPPO-Learned
```python
LSSL(measure='legs', learn=1)
```
- HiPPO (LegS) initialization
- **Trainable** A matrix
- **Use case**: Best of both (structure + learning)

---

## Project Structure

```
SC/
├── data/dataset.py           # SPEECHCOMMANDS dataset
├── features/mfcc.py          # MFCC extraction
├── models/
│   ├── lstm.py               # LSTM model
│   ├── transformer.py        # Transformer model
│   ├── lssl.py              # LSSL model (uses official implementation)
│   └── conv_frontend.py      # 1D conv features
├── s4/                       # Official S4 repo (contains LSSL implementation)
├── train.py                  # Main training script
├── config.yaml               # Configuration
└── setup.sh                  # Environment setup
```

---

## Key Files

- **README.md**: Full user guide
- **LSSL_VS_S4.md**: Detailed comparison of LSSL, S4, S4D
- **IMPLEMENTATION.md**: Technical implementation details
- **PROJECT_SUMMARY.md**: Complete project overview
- **QUICK_REFERENCE.md**: This file - quick lookup

---

## Common Questions

### Q: Why not use S4 or S4D directly?
**A:** We use LSSL as the baseline. S4 and S4D can be added later as **separate models** for comparison.

### Q: Is LSSL slower than S4/S4D?
**A:** Yes, LSSL uses Krylov methods which are slower. S4 (2022) improved this with DPLR parameterization.

### Q: Should I use HiPPO-fixed or HiPPO-learned?
**A:** Usually **HiPPO-learned**. Fixed is mainly for ablation studies.

### Q: What's the difference between measures (LegS, LegT, etc.)?
**A:** Different HiPPO polynomial bases:
- **LegS**: Legendre (Scaled) - most common
- **LegT**: Legendre (Translated) - alternative
- **LagT**: Laguerre - for unbounded memory

### Q: How do I add S4 or S4D?
**A:** See [LSSL_VS_S4.md](LSSL_VS_S4.md) for instructions.

---

## Outputs

Each experiment produces:
- `logs/{exp_name}_best.pth` - Best model checkpoint
- `logs/{exp_name}_metrics.csv` - Metrics (spreadsheet-ready)
- `logs/{exp_name}_metrics.json` - Metrics (programmatic)
- `runs/{exp_name}/` - TensorBoard events

---

## Typical Workflow

1. **Setup**: `bash setup.sh`
2. **Test**: `python test_setup.py`
3. **Run baseline**: `python train.py --model lstm --input_type raw`
4. **Run LSSL**: `python train.py --model lssl --input_type raw --lssl_variant hippo_learned`
5. **Compare**: `tensorboard --logdir runs`
6. **Analyze**: Check `logs/*.csv` files
7. **Extend**: Add S4/S4D as separate models
