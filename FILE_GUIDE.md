# Documentation File Guide

## 📖 Which File Should I Read?

### Quick Start
- **README_FIRST.md** ← **START HERE**
  - 3-command quick start
  - Common questions answered
  - Next steps guide

### Usage
- **README.md**
  - Complete user guide
  - All commands and examples
  - Installation instructions

- **QUICK_REFERENCE.md**
  - Command cheat sheet
  - Quick lookup
  - Common patterns

### Understanding LSSL
- **LSSL_VS_S4.md** ← **IMPORTANT**
  - What is LSSL? (vs S4 vs S4D)
  - Timeline and history
  - How to add S4/S4D later
  - **Read this to understand the models!**

### Implementation Details
- **IMPLEMENTATION.md**
  - Technical deep-dive
  - Architecture details
  - How each model works

- **PROJECT_SUMMARY.md**
  - Complete project overview
  - All 9 configurations
  - Design decisions

### Changes Made
- **CORRECTIONS.md**
  - What was fixed
  - Why it matters
  - Before/after code

## 📁 File Organization

```
Documentation:
├── README_FIRST.md       ← Start here
├── README.md             ← Full guide
├── QUICK_REFERENCE.md    ← Command cheat sheet
├── LSSL_VS_S4.md        ← Model comparison (IMPORTANT)
├── IMPLEMENTATION.md     ← Technical details
├── PROJECT_SUMMARY.md    ← Complete overview
├── CORRECTIONS.md        ← What was fixed
└── FILE_GUIDE.md         ← This file

Code:
├── data/
│   └── dataset.py        ← SPEECHCOMMANDS loader
├── features/
│   └── mfcc.py          ← MFCC extraction
├── models/
│   ├── lssl.py          ← LSSL model (uses official LSSL!)
│   ├── lstm.py          ← LSTM baseline
│   ├── transformer.py   ← Transformer baseline
│   └── conv_frontend.py ← 1D conv features
├── s4/                  ← Official S4 repo (contains LSSL)
├── train.py             ← Main script
├── utils.py             ← Training utilities
├── config.yaml          ← Configuration
├── test_setup.py        ← Verify installation
├── setup.sh             ← Environment setup
└── run_all_experiments.sh ← Run all 9 configs

Outputs (generated):
├── logs/                ← Checkpoints and metrics
├── runs/                ← TensorBoard events
└── venv/                ← Virtual environment
```

## 🎯 Reading Path by Goal

### Goal: Just run experiments
1. README_FIRST.md (quick start)
2. QUICK_REFERENCE.md (commands)

### Goal: Understand LSSL
1. LSSL_VS_S4.md (what is LSSL?)
2. IMPLEMENTATION.md (how does it work?)

### Goal: Extend the project
1. IMPLEMENTATION.md (technical details)
2. LSSL_VS_S4.md (how to add S4/S4D)
3. PROJECT_SUMMARY.md (design decisions)

### Goal: Debug issues
1. CORRECTIONS.md (what changed?)
2. IMPLEMENTATION.md (how it works)
3. test_setup.py (run tests)

## 🔑 Key Takeaways

**Most important files:**
1. **README_FIRST.md** - Your starting point
2. **LSSL_VS_S4.md** - Understand what LSSL is (not S4/S4D!)
3. **QUICK_REFERENCE.md** - Command reference

**One-sentence summary:**
> This project uses **LSSL** (NeurIPS 2021), the original state-space layer. S4 and S4D are different, newer models that can be added later for comparison.
