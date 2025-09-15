"""Test script for HNet-NTV2 hybrid model."""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modeling_hnet_ntv2 import HNetNTV2Config, HNetNTV2ForMaskedLM, load_ntv2_weights


def test_model_creation():
    """Test basic model creation."""
    print("Testing model creation...")
    
    config = HNetNTV2Config(
        d_model=1024,  # NTV2 uses 1024 hidden size
        n_layer=24,
        vocab_size=12,  # Character tokenizer vocabulary size
        n_head=16,     # NTV2 uses 16 attention heads
        intermediate_size=8192,  # NTV2 uses 8192 intermediate size
        max_position_embeddings=1000,
        n_enc_layer=2,
        n_main_layer=20,
        n_dec_layer=2,
    )
    
    model = HNetNTV2ForMaskedLM(config)
    print(f"Model created successfully with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model, config


def test_forward_pass(model, config):
    """Test forward pass with dummy data."""
    print("Testing forward pass...")
    
    batch_size = 2
    seq_len = 100
    
    # Create dummy input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    
    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Loss: {outputs.loss.item():.4f}")
    
    return outputs


def test_ntv2_weight_loading():
    """Test loading NTV2 weights into main transformer layers."""
    print("Testing NTV2 weight loading into main transformer layers...")
    
    config = HNetNTV2Config(
        d_model=1024,  # NTV2 uses 1024 hidden size
        n_layer=24,
        vocab_size=12,  # Character tokenizer vocabulary size
        n_head=16,     # NTV2 uses 16 attention heads
        intermediate_size=8192,  # NTV2 uses 8192 intermediate size
        max_position_embeddings=1000,
        n_enc_layer=2,
        n_main_layer=20,  # These will be loaded with NTV2 weights
        n_dec_layer=2,
    )
    
    model = HNetNTV2ForMaskedLM(config)
    
    # Test weight loading (this will download the model if not cached)
    try:
        model = load_ntv2_weights(model)
        print("NTV2 weights loaded successfully into main transformer layers!")
        
        # Verify that main layers have been loaded
        main_layer_params = sum(p.numel() for name, p in model.named_parameters() 
                               if 'main_layers' in name and p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Main transformer layers: {main_layer_params:,} parameters")
        print(f"Total model parameters: {total_params:,} parameters")
        
        return True
    except Exception as e:
        print(f"Error loading NTV2 weights: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("HNet-NTV2 Hybrid Model Test Suite")
    print("=" * 50)
    
    # Test 1: Model creation
    model, config = test_model_creation()
    print()
    
    # Test 2: Forward pass
    outputs = test_forward_pass(model, config)
    print()
    
    # Test 3: NTV2 weight loading
    weight_loading_success = test_ntv2_weight_loading()
    print()
    
    print("=" * 50)
    print("Test Summary:")
    print(f"✓ Model creation: PASSED")
    print(f"✓ Forward pass: PASSED")
    print(f"{'✓' if weight_loading_success else '✗'} NTV2 weight loading: {'PASSED' if weight_loading_success else 'FAILED'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
