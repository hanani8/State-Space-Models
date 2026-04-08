#!/bin/bash

# Script to run all 9 model configurations with PyTorch Lightning
# Automatically uses all available GPUs
# Usage: bash run_all_experiments_lightning.sh

echo "Starting speech classification benchmarks (PyTorch Lightning + Multi-GPU)..."
echo "==============================================================================="

# Auto-detect number of GPUs
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected $NUM_GPUS GPU(s)"
echo ""

# LSTM experiments
echo -e "\n[1/9] LSTM + Raw Waveform"
python train_lightning.py --model lstm --input_type raw --epochs 30 --gpus $NUM_GPUS

echo -e "\n[2/9] LSTM + Conv Frontend"
python train_lightning.py --model lstm --input_type conv --epochs 30 --gpus $NUM_GPUS

echo -e "\n[3/9] LSTM + MFCC"
python train_lightning.py --model lstm --input_type mfcc --epochs 30 --gpus $NUM_GPUS

# Transformer experiments
echo -e "\n[4/9] Transformer + Raw Waveform"
python train_lightning.py --model transformer --input_type raw --epochs 30 --gpus $NUM_GPUS

echo -e "\n[5/9] Transformer + Conv Frontend"
python train_lightning.py --model transformer --input_type conv --epochs 30 --gpus $NUM_GPUS

echo -e "\n[6/9] Transformer + MFCC"
python train_lightning.py --model transformer --input_type mfcc --epochs 30 --gpus $NUM_GPUS

# LSSL experiments
echo -e "\n[7/9] LSSL Vanilla (Random Init)"
python train_lightning.py --model lssl --input_type raw --lssl_variant vanilla --epochs 30 --gpus $NUM_GPUS

echo -e "\n[8/9] LSSL HiPPO Fixed"
python train_lightning.py --model lssl --input_type raw --lssl_variant hippo_fixed --epochs 30 --gpus $NUM_GPUS

echo -e "\n[9/9] LSSL HiPPO Learned"
python train_lightning.py --model lssl --input_type raw --lssl_variant hippo_learned --epochs 30 --gpus $NUM_GPUS

echo -e "\n==============================================================================="
echo "All experiments completed!"
echo "View results with: tensorboard --logdir runs"
