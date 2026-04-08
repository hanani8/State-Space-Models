# LSSL vs S4 vs S4D: Understanding the Differences

## What This Project Uses

This benchmark uses **LSSL** (NeurIPS 2021), the **original** Linear State-Space Layer.

**S4** and **S4D** are improvements that came later and can be added as separate models for comparison.

---

## Timeline and Relationships

```
2020: HiPPO (NeurIPS)
      └─ Theory for continuous-time memorization
         using orthogonal polynomials

2021: LSSL (NeurIPS) ← WE USE THIS
      └─ First trainable state-space layer
         Built on HiPPO
         Has speed/memory issues
         Foundation for later work

2022: S4 (ICLR) ← FUTURE: Add as separate model
      └─ Efficient version of LSSL
         DPLR parameterization (Diagonal + Low-Rank)
         Fast Cauchy kernel
         Much faster and more scalable

2022: S4D (NeurIPS) ← FUTURE: Add as separate model
      └─ Simplified S4
         Diagonal-only (no low-rank component)
         Easier to understand
         Nearly as effective as S4
```

---

## LSSL (What We're Using)

### Paper
> **Combining Recurrent, Convolutional, and Continuous-time Models with Linear State Space Layers**  
> Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, Christopher Ré  
> NeurIPS 2021

### Key Characteristics

**State-Space Formulation:**
```
x'(t) = Ax(t) + Bu(t)    (continuous time)
x_k = A̅x_{k-1} + B̅u_k   (discrete time, after discretization)
y_k = Cx_k + Du_k
```

**Matrix Structure:**
- A can be:
  - **Random**: Randomly initialized (vanilla variant)
  - **HiPPO**: Structured using LegS, LegT, or other HiPPO matrices (HiPPO variants)
- Full N×N matrix (not restricted to diagonal)

**Training:**
- Uses Krylov methods to compute convolution kernel
- Can be frozen (learn=0) or trainable (learn=1 or 2)

**Advantages:**
- **Theoretically grounded** via HiPPO
- **Flexible** matrix structure
- **Foundation** for later improvements

**Limitations (from the S4 paper):**
- Slower than S4 due to Krylov computation
- Memory issues on very long sequences
- Didn't fully leverage convolution/recurrence duality

### Our Three Variants

| Variant | Measure | Learn | Description |
|---------|---------|-------|-------------|
| **Vanilla** | `'random'` | `1` | Random A matrix, trainable |
| **HiPPO-Fixed** | `'legs'` | `0` | HiPPO (LegS) init, frozen |
| **HiPPO-Learned** | `'legs'` | `1` | HiPPO (LegS) init, trainable |

**HiPPO Measures:**
- `'legs'`: Legendre (Scaled) - bounded memory
- `'legt'`: Legendre (Translated) - bounded memory, different scaling
- `'lagt'`: Laguerre (Translated) - unbounded memory

We use **LegS** as it's one of the most common and effective.

---

## S4 (Future Addition)

### Paper
> **Efficiently Modeling Long Sequences with Structured State Spaces**  
> Albert Gu, Karan Goel, Christopher Ré  
> ICLR 2022 (Outstanding Paper Honorable Mention)

### Key Improvements Over LSSL

**DPLR Parameterization:**
```
A = Λ - PP*

Where:
- Λ: Diagonal (complex)
- P: Low-rank (R×N), typically R=1 or 2
- PP*: Low-rank correction
```

This structure:
- Enables fast **Cauchy kernel** evaluation
- Reduces computation from O(N²L) to O(NL log L)
- Maintains expressiveness of full matrix

**Discretization:**
- More sophisticated bilinear transform
- Better numerical properties

**Speed:**
- 10-100× faster than LSSL on long sequences
- Optional CUDA kernels for further speedup

### When to Add S4

Add S4 to your benchmark when you want to:
- Compare modern efficient SSMs vs original LSSL
- Understand the impact of DPLR parameterization
- Test on very long sequences (>2000 steps)
- Benchmark training speed

**How to add:**
```python
from models.s4.s4 import S4Block

layer = S4Block(
    d_model=128,
    d_state=64,
    mode='dplr',
    init='hippo',
    dropout=0.1
)
```

---

## S4D (Future Addition)

### Paper
> **On the Parameterization and Initialization of Diagonal State Space Models**  
> Albert Gu, Ankit Gupta, Karan Goel, Christopher Ré  
> NeurIPS 2022

### Key Simplification

**Diagonal-Only:**
```
A = Λ  (just diagonal, no low-rank part)

Where:
- Λ_real: Decay rates (negative)
- Λ_imag: Frequencies (controls oscillation)
```

**Initialization:**
- Can use modified HiPPO (S4D-LegS)
- Or simple heuristics (S4D-Lin, S4D-Inv)

**Advantages:**
- **Simpler** than S4 (no low-rank component)
- **Faster** than S4 (diagonal operations)
- **Fewer parameters**
- **Nearly as effective** as S4 in practice

### When to Add S4D

Add S4D when you want to:
- Find the simplest effective SSM
- Reduce parameter count
- Maximize training/inference speed
- Test if low-rank component is necessary

**How to add:**
```python
from models.s4.s4d import S4D

layer = S4D(
    d_model=128,
    d_state=64,
    dropout=0.1,
    transposed=True
)
```

---

## Comparison Table

| Feature | LSSL (We use this) | S4 (Add later) | S4D (Add later) |
|---------|-------------------|----------------|-----------------|
| **Year** | 2021 | 2022 | 2022 |
| **A Matrix** | Full N×N | Diagonal + Low-Rank | Diagonal only |
| **Init Options** | Random, HiPPO | HiPPO (DPLR) | Modified HiPPO, heuristics |
| **Computation** | Krylov | Cauchy kernel | Vandermonde |
| **Speed** | Slower | Fast | Fastest |
| **Memory** | Higher | Lower | Lowest |
| **Parameters** | O(N²) or O(N) | O(N) | O(N) |
| **Complexity** | O(N²L) | O(NL log L) | O(NL log L) |
| **Use Case** | Original/baseline | Long sequences | Simple/efficient |

---

## Why Start with LSSL?

1. **Historical completeness**: LSSL is where it all started
2. **Fair comparison**: Shows evolution of the field
3. **Ablation study**: Can compare improvements of S4/S4D over baseline
4. **Flexibility**: Full matrix gives maximum expressiveness
5. **Your spec**: You explicitly requested LSSL as separate from S4/S4D

---

## Research Questions You Can Answer

### Current (with LSSL):
- Does HiPPO initialization help? (fixed vs learned)
- Is random init competitive?
- How does LSSL compare to LSTM and Transformer?

### After adding S4:
- How much does DPLR improve over full matrix?
- Is S4 faster than LSSL in practice?
- Does low-rank structure help?

### After adding S4D:
- Is diagonal sufficient for speech?
- Can we simplify further without losing performance?
- What's the speed/accuracy tradeoff?

---

## Implementation Status

### ✅ Implemented (Current)
- [x] LSSL with random initialization (vanilla)
- [x] LSSL with HiPPO init, frozen (hippo_fixed)
- [x] LSSL with HiPPO init, trainable (hippo_learned)

### 🚧 Future Work
- [ ] S4 with DPLR parameterization
- [ ] S4D with diagonal parameterization
- [ ] Comparison across all 5 SSM variants
- [ ] Speed benchmarks

---

## How to Extend

### Adding S4 as a Separate Model

1. Create `models/s4_model.py`:
```python
from models.s4.s4 import S4Block

class S4Model(nn.Module):
    # Similar structure to LSSLModel
    # Use S4Block instead of LSSL
    ...
```

2. Register in `train.py`:
```python
elif model_type == 's4':
    model = S4Model(...)
```

3. Run comparison:
```bash
python train.py --model lssl --input_type raw --lssl_variant hippo_learned
python train.py --model s4 --input_type raw
```

### Adding S4D as a Separate Model

Same process, but use `S4D` instead of `S4Block`.

---

## Summary

**Current state:**
- Using LSSL (NeurIPS 2021) - the original state-space layer
- Three variants to test initialization strategies
- Clean baseline for future comparisons

**Future state:**
- Add S4 to test improved parameterization
- Add S4D to test diagonal simplification
- Compare all SSM variants head-to-head

This gives you the complete picture of state-space model evolution!
