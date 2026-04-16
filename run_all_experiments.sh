#!/bin/bash
# Core comparison: LSTM vs Transformer vs S4D vs S4 across input representations.
# Usage: bash run_all_experiments.sh [--epochs N]

EPOCHS=30
if [[ "$1" == "--epochs" ]]; then
    EPOCHS=$2
fi

echo "Starting core comparison benchmarks (epochs=$EPOCHS)"
echo "======================================================"

# ── LSTM ──────────────────────────────────────────────────────────────────────
echo -e "\n[1/11] LSTM + Raw Waveform"
conda run -n ssm python train.py --model lstm --input_type raw --epochs $EPOCHS

echo -e "\n[2/11] LSTM + Conv Frontend"
conda run -n ssm python train.py --model lstm --input_type conv --epochs $EPOCHS

echo -e "\n[3/11] LSTM + MFCC"
conda run -n ssm python train.py --model lstm --input_type mfcc --epochs $EPOCHS

# ── Transformer ───────────────────────────────────────────────────────────────
echo -e "\n[4/11] Transformer + Raw Waveform"
conda run -n ssm python train.py --model transformer --input_type raw --epochs $EPOCHS

echo -e "\n[5/11] Transformer + Conv Frontend"
conda run -n ssm python train.py --model transformer --input_type conv --epochs $EPOCHS

echo -e "\n[6/11] Transformer + MFCC"
conda run -n ssm python train.py --model transformer --input_type mfcc --epochs $EPOCHS

# ── S4D (diagonal SSM, NeurIPS 2022) ─────────────────────────────────────────
echo -e "\n[7/11] S4D + Raw Waveform  (diagonal, N=64)"
conda run -n ssm python train.py --model s4 --input_type raw  --ssm_type s4d --epochs $EPOCHS

echo -e "\n[8/11] S4D + Conv Frontend"
conda run -n ssm python train.py --model s4 --input_type conv --ssm_type s4d --epochs $EPOCHS

echo -e "\n[9/11] S4D + MFCC"
conda run -n ssm python train.py --model s4 --input_type mfcc --ssm_type s4d --epochs $EPOCHS

# ── S4 (NPLR + HiPPO-LegS, ICLR 2022) ───────────────────────────────────────
# Raw waveform is the most meaningful comparison: L=16000 is where S4's
# structured kernel (vs Transformer's O(L²) attention) matters most.
echo -e "\n[10/11] S4 + Raw Waveform  (NPLR/HiPPO-LegS, N=64)"
conda run -n ssm python train.py --model s4 --input_type raw  --ssm_type s4 --epochs $EPOCHS

echo -e "\n[11/11] S4 + MFCC  (NPLR/HiPPO-LegS, N=64)"
conda run -n ssm python train.py --model s4 --input_type mfcc --ssm_type s4 --epochs $EPOCHS

echo -e "\n======================================================"
echo "Core comparison complete."
echo "View results: tensorboard --logdir runs"
