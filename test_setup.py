"""
Test script to verify setup and imports.
"""
import sys
import torch

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from data.dataset import get_dataloaders
        print("✓ Data module")
    except Exception as e:
        print(f"✗ Data module: {e}")
        return False

    try:
        from features.mfcc import MFCCTransform
        print("✓ Features module")
    except Exception as e:
        print(f"✗ Features module: {e}")
        return False

    try:
        from models.lstm import LSTMModel
        from models.transformer import TransformerModel
        from models.conv_frontend import Conv1DFrontend
        print("✓ Basic models (LSTM, Transformer, Conv)")
    except Exception as e:
        print(f"✗ Basic models: {e}")
        return False

    try:
        from models.lssl import LSSLModel
        print("✓ LSSL model (S4 integration)")
    except Exception as e:
        print(f"✗ LSSL model: {e}")
        print("  Make sure S4 repo is cloned and dependencies installed")
        return False

    try:
        from utils import setup_optimizer, MetricsLogger, evaluate
        print("✓ Utilities")
    except Exception as e:
        print(f"✗ Utilities: {e}")
        return False

    return True


def test_model_creation():
    """Test that models can be instantiated."""
    print("\nTesting model instantiation...")

    device = 'cpu'
    num_classes = 10
    batch_size = 4
    seq_len = 100

    try:
        from models.lstm import LSTMModel
        model = LSTMModel(input_dim=40, num_classes=num_classes)
        x = torch.randn(batch_size, seq_len, 40)
        y = model(x)
        assert y.shape == (batch_size, num_classes)
        print("✓ LSTM model")
    except Exception as e:
        print(f"✗ LSTM model: {e}")
        return False

    try:
        from models.transformer import TransformerModel
        model = TransformerModel(input_dim=40, num_classes=num_classes)
        x = torch.randn(batch_size, seq_len, 40)
        y = model(x)
        assert y.shape == (batch_size, num_classes)
        print("✓ Transformer model")
    except Exception as e:
        print(f"✗ Transformer model: {e}")
        return False

    try:
        from models.lssl import LSSLModel
        # Test vanilla variant
        model = LSSLModel(input_dim=40, num_classes=num_classes, lssl_variant='vanilla')
        x = torch.randn(batch_size, seq_len, 40)
        y = model(x)
        assert y.shape == (batch_size, num_classes)
        print("✓ LSSL model (vanilla)")
    except Exception as e:
        print(f"✗ LSSL model: {e}")
        return False

    try:
        from models.conv_frontend import Conv1DFrontend
        frontend = Conv1DFrontend()
        x = torch.randn(batch_size, seq_len)
        y = frontend(x)
        print(f"✓ Conv frontend (output shape: {y.shape})")
    except Exception as e:
        print(f"✗ Conv frontend: {e}")
        return False

    try:
        from features.mfcc import MFCCTransform
        mfcc = MFCCTransform()
        # Use realistic audio length (16000 samples = 1 second at 16kHz)
        x = torch.randn(batch_size, 16000)
        y = mfcc(x)
        print(f"✓ MFCC transform (output shape: {y.shape})")
    except Exception as e:
        print(f"✗ MFCC transform: {e}")
        return False

    return True


def main():
    print("="*60)
    print("Speech Classification Benchmark - Setup Test")
    print("="*60)

    print(f"\nPython version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not test_imports():
        print("\n❌ Import test failed!")
        return 1

    if not test_model_creation():
        print("\n❌ Model creation test failed!")
        return 1

    print("\n" + "="*60)
    print("✅ All tests passed! Setup is complete.")
    print("="*60)
    print("\nYou can now run experiments with:")
    print("  python train.py --model lstm --input_type raw")
    print("  python train.py --model transformer --input_type mfcc")
    print("  python train.py --model lssl --input_type raw --lssl_variant hippo_learned")
    print("\nOr run all experiments with:")
    print("  bash run_all_experiments.sh")

    return 0


if __name__ == '__main__':
    sys.exit(main())
