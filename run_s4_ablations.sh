#!/bin/bash
# S4 ablation study — all runs use raw waveform (L=16 000 at 16 kHz) to isolate
# SSM-specific hyper-parameters without input-representation confounds.
#
# Usage: bash run_s4_ablations.sh [--epochs N]

EPOCHS=30
if [[ "$1" == "--epochs" ]]; then
    EPOCHS=$2
fi

echo "Starting S4 ablation study (epochs=$EPOCHS)"
echo "============================================="

# ── 1. S4D vs S4 (kernel architecture) ───────────────────────────────────────
# Controls everything else: same d_model=128, d_state=64, 4 layers, raw audio.
# Question: does the NPLR + HiPPO-LegS init of full S4 improve over S4D?
echo -e "\n[1/8] S4D  raw  N=64  (diagonal, NeurIPS 2022)"
conda run -n ssm python train.py --model s4 --input_type raw --ssm_type s4d --d_state 64 --epochs $EPOCHS

echo -e "\n[2/8] S4   raw  N=64  (NPLR/HiPPO-LegS, ICLR 2022)"
conda run -n ssm python train.py --model s4 --input_type raw --ssm_type s4  --d_state 64 --epochs $EPOCHS

# ── 2. State size N (S4D) ─────────────────────────────────────────────────────
# N controls how many complex frequencies the SSM kernel represents.
# Use S4D for speed; the trend should hold for full S4 too.
# Question: is N=64 a sweet spot, or does more state help for raw speech?
echo -e "\n[3/8] S4D  raw  N=32   (small state)"
conda run -n ssm python train.py --model s4 --input_type raw --ssm_type s4d --d_state 32  --epochs $EPOCHS

echo -e "\n[4/8] S4D  raw  N=128"
conda run -n ssm python train.py --model s4 --input_type raw --ssm_type s4d --d_state 128 --epochs $EPOCHS

echo -e "\n[5/8] S4D  raw  N=256  (large state)"
conda run -n ssm python train.py --model s4 --input_type raw --ssm_type s4d --d_state 256 --epochs $EPOCHS

# ── 3. Depth (S4D, N=64) ──────────────────────────────────────────────────────
# More layers = deeper temporal hierarchy; fewer = faster convergence.
# Question: does stacking more S4D blocks help on 1-second speech clips?
echo -e "\n[6/8] S4D  raw  2 layers"
conda run -n ssm python train.py --model s4 --input_type raw --ssm_type s4d --num_layers 2 --epochs $EPOCHS

# N=64, 4 layers is [1/8] above — no need to repeat.

echo -e "\n[7/8] S4D  raw  6 layers"
conda run -n ssm python train.py --model s4 --input_type raw --ssm_type s4d --num_layers 6 --epochs $EPOCHS

# ── 4. State size N (S4, N=256) ───────────────────────────────────────────────
# Verify whether the full S4 kernel benefits more from larger state than S4D.
echo -e "\n[8/8] S4   raw  N=256  (NPLR/HiPPO-LegS, large state)"
conda run -n ssm python train.py --model s4 --input_type raw --ssm_type s4 --d_state 256 --epochs $EPOCHS

echo -e "\n============================================="
echo "S4 ablation study complete."
echo "View results: tensorboard --logdir runs"
