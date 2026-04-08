# 🎯 Start Here

## What You Have

A **complete, modular PyTorch benchmark** for comparing sequence models on speech classification.

### ✅ Implemented (Ready to Use)

**9 Model Configurations:**
- LSTM: raw / conv / mfcc
- Transformer: raw / conv / mfcc  
- **LSSL**: vanilla / hippo_fixed / hippo_learned

**Note:** LSSL = Linear State-Space Layer (NeurIPS 2021), the **original** SSM layer.  
S4 and S4D are **different models** that can be added later for comparison.

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Setup environment
bash setup.sh

# 2. Test setup
source venv/bin/activate
python test_setup.py

# 3. Run an experiment
python train.py --model lssl --input_type raw --lssl_variant hippo_learned
```

---

## 📚 Documentation Guide

**New user? Read in this order:**

1. **README_FIRST.md** (this file) - Start here
2. **README.md** - Full user guide with all commands
3. **QUICK_REFERENCE.md** - Command cheat sheet
4. **LSSL_VS_S4.md** - Understand LSSL vs S4 vs S4D

**Implementing or extending? Read these:**

5. **IMPLEMENTATION.md** - Technical deep-dive
6. **PROJECT_SUMMARY.md** - Complete overview

**What changed?**

7. **CORRECTIONS.md** - Explains the LSSL vs S4/S4D fix

---

## ❓ Common Questions

### Q: What's LSSL?
**A:** Linear State-Space Layer (NeurIPS 2021). The **original** trainable state-space model. Came **before** S4 and S4D.

### Q: Is this S4 or S4D?
**A:** **No.** This uses LSSL. S4 and S4D are **different, newer models** (2022) that can be added later.

### Q: Can I compare LSSL vs S4 vs S4D?
**A:** Yes! LSSL is implemented now. Add S4 and S4D later as **separate models**. See `LSSL_VS_S4.md` for instructions.

### Q: Which LSSL variant should I use?
**A:** Usually `hippo_learned` (HiPPO init + trainable). Use others for ablation studies.

### Q: What's HiPPO?
**A:** Theoretically-motivated initialization using orthogonal polynomials (Legendre, Laguerre). Enables long-range memory.

---

## 🎯 Your Next Steps

### 1. Verify Setup Works
```bash
source venv/bin/activate
python test_setup.py
```

Expected output: `✅ All tests passed!`

### 2. Run a Quick Experiment (~10 min)
```bash
python train.py --model lstm --input_type raw --epochs 5
```

This runs a short LSTM baseline to verify everything works.

### 3. Run Full LSSL Comparison (~1 hour)
```bash
# Vanilla (random init)
python train.py --model lssl --input_type raw --lssl_variant vanilla

# HiPPO-fixed (structured, frozen)
python train.py --model lssl --input_type raw --lssl_variant hippo_fixed

# HiPPO-learned (structured, trainable) 
python train.py --model lssl --input_type raw --lssl_variant hippo_learned
```

### 4. View Results
```bash
tensorboard --logdir runs
```

Open http://localhost:6006 in your browser.

### 5. Run All 9 Configurations (~3-5 hours)
```bash
bash run_all_experiments.sh
```

---

## 📊 What You'll Learn

From the current implementation:
- How does LSSL compare to LSTM and Transformer?
- Does HiPPO initialization help?
- Is structure alone enough (fixed) or do we need learning?
- Which input type works best (raw/conv/mfcc)?

Future (after adding S4/S4D):
- How much faster is S4 than LSSL?
- Is the low-rank DPLR parameterization necessary?
- Can we simplify to diagonal-only (S4D)?

---

## 🔧 Project Structure

```
SC/
├── data/dataset.py         # SPEECHCOMMANDS dataset
├── models/
│   ├── lssl.py            # ✅ Uses actual LSSL (NeurIPS 2021)
│   ├── lstm.py            # LSTM baseline
│   └── transformer.py     # Transformer baseline
├── s4/                    # Official repo (contains LSSL implementation)
├── train.py               # Main training script
└── config.yaml            # Configuration
```

---

## 🎓 Key Papers

**What this project uses:**

```bibtex
@article{gu2021combining,
  title={Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers},
  author={Gu, Albert and Johnson, Isys and Goel, Karan and Saab, Khaled and Dao, Tri and Rudra, Atri and R{\'e}, Christopher},
  journal={NeurIPS},
  year={2021}
}
```

**Foundation:**

```bibtex
@article{gu2020hippo,
  title={HiPPO: Recurrent Memory with Optimal Polynomial Projections},
  author={Gu, Albert and Dao, Tri and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  journal={NeurIPS},
  year={2020}
}
```

---

## ✨ Ready to Go!

The project is fully functional. Just run:

```bash
bash setup.sh && source venv/bin/activate && python test_setup.py
```

If all tests pass, you're ready to start benchmarking!

For detailed instructions, see **README.md**.  
For quick command reference, see **QUICK_REFERENCE.md**.  
For LSSL vs S4 vs S4D, see **LSSL_VS_S4.md**.

---

**Questions?** Check the documentation files above.  
**Issues?** Make sure you've activated the virtual environment: `source venv/bin/activate`
