#!/bin/bash

# Setup script for Speech Classification Benchmark
# Uses conda environment 'ssm'

echo "Setting up Speech Classification Benchmark..."
echo "=============================================="

# Activate conda environment
echo "Activating conda environment 'ssm'..."
conda activate ssm

# Check if any additional dependencies are missing
echo "Checking dependencies..."

# Install any missing S4 dependencies that aren't already in conda env
echo "Checking S4 dependencies..."
cd s4
# Most dependencies should already be in ssm env, but check anyway
pip install -r requirements.txt --no-deps 2>/dev/null || true
cd ..

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate ssm"
echo ""
echo "To test the setup, run:"
echo "  python test_setup.py"
echo ""
echo "To start training, run:"
echo "  python train.py --model lstm --input_type raw"
