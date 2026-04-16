#!/bin/bash
# Core comparison with PyTorch Lightning (multi-GPU support).
# Usage: bash run_all_experiments_lightning.sh [--epochs N]

EPOCHS=30
if [[ "$1" == "--epochs" ]]; then
    EPOCHS=$2
fi

echo "Starting speech classification benchmarks (PyTorch Lightning, epochs=$EPOCHS)"
echo "==============================================================================="

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected $NUM_GPUS GPU(s)"
echo ""

# ── LSTM ──────────────────────────────────────────────────────────────────────
echo -e "\n[1/9] LSTM + Raw Waveform"
python train_lightning.py --model lstm --input_type raw --epochs $EPOCHS --gpus $NUM_GPUS

echo -e "\n[2/9] LSTM + Conv Frontend"
python train_lightning.py --model lstm --input_type conv --epochs $EPOCHS --gpus $NUM_GPUS

echo -e "\n[3/9] LSTM + MFCC"
python train_lightning.py --model lstm --input_type mfcc --epochs $EPOCHS --gpus $NUM_GPUS

# ── Transformer ───────────────────────────────────────────────────────────────
echo -e "\n[4/9] Transformer + Raw Waveform"
python train_lightning.py --model transformer --input_type raw --epochs $EPOCHS --gpus $NUM_GPUS

echo -e "\n[5/9] Transformer + Conv Frontend"
python train_lightning.py --model transformer --input_type conv --epochs $EPOCHS --gpus $NUM_GPUS

echo -e "\n[6/9] Transformer + MFCC"
python train_lightning.py --model transformer --input_type mfcc --epochs $EPOCHS --gpus $NUM_GPUS

# ── S4D (diagonal SSM, NeurIPS 2022) ─────────────────────────────────────────
echo -e "\n[7/9] S4D + Raw Waveform  (diagonal, N=64)"
conda run -n ssm python train_lightning.py --model s4 --input_type raw  --ssm_type s4d --epochs $EPOCHS --gpus $NUM_GPUS

echo -e "\n[8/9] S4D + Conv Frontend"
conda run -n ssm python train_lightning.py --model s4 --input_type conv --ssm_type s4d --epochs $EPOCHS --gpus $NUM_GPUS

echo -e "\n[9/9] S4D + MFCC"
conda run -n ssm python train_lightning.py --model s4 --input_type mfcc --ssm_type s4d --epochs $EPOCHS --gpus $NUM_GPUS

# ── S4 (NPLR + HiPPO-LegS, ICLR 2022) ───────────────────────────────────────
echo -e "\n[10/11] S4 + Raw Waveform  (NPLR/HiPPO-LegS, N=64)"
conda run -n ssm python train_lightning.py --model s4 --input_type raw  --ssm_type s4 --epochs $EPOCHS --gpus $NUM_GPUS

echo -e "\n[11/11] S4 + MFCC"
conda run -n ssm python train_lightning.py --model s4 --input_type mfcc --ssm_type s4 --epochs $EPOCHS --gpus $NUM_GPUS

echo -e "\n==============================================================================="
echo "Core comparison complete."
echo "View results: tensorboard --logdir runs"
