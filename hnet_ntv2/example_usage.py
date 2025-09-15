"""Example usage of HNet-NTV2 hybrid model."""

import torch
from modeling_hnet_ntv2 import HNetNTV2Config, HNetNTV2ForMaskedLM, load_ntv2_weights


def create_model_with_ntv2_weights():
    """Create the hybrid model and load NTV2 weights into the main transformer."""
    print("Creating HNet-NTV2 hybrid model...")
    
    # Configure the model to match NTV2 architecture
    config = HNetNTV2Config(
        d_model=512,           # NTV2 hidden size
        n_layer=24,            # NTV2 number of layers
        vocab_size=4105,       # NTV2 vocabulary size (6-mer tokenizer)
        n_head=8,              # NTV2 attention heads
        intermediate_size=2048, # NTV2 MLP intermediate size
        max_position_embeddings=1000,  # NTV2 max sequence length
        
        # HNet-specific configuration
        n_enc_layer=2,         # Encoder layers for chunking
        n_main_layer=20,       # Main transformer layers (will be loaded with NTV2 weights)
        n_dec_layer=2,         # Decoder layers for dechunking
        
        # Tokenization
        tokenizer_type="hnet", # Use HNet-style tokenization
        use_rotary_embeddings=True,  # NTV2 uses rotary embeddings
        
        # Training parameters
        target_ratio=0.25,     # HNet chunking ratio
    )
    
    # Create the model
    model = HNetNTV2ForMaskedLM(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load NTV2 weights into the main transformer layers
    print("\nLoading NTV2 weights into main transformer layers...")
    model = load_ntv2_weights(model)
    
    return model, config


def demonstrate_forward_pass(model, config):
    """Demonstrate forward pass with DNA sequences."""
    print("\nDemonstrating forward pass...")
    
    # Create dummy DNA sequences (using NTV2 vocabulary)
    batch_size = 2
    seq_len = 100
    
    # Random DNA token IDs (simulating 6-mer tokenization)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Create some labels for masked language modeling
    labels = input_ids.clone()
    # Mask some tokens (set to -100 to ignore in loss)
    mask_indices = torch.rand(batch_size, seq_len) < 0.15  # 15% masking
    labels[~mask_indices] = -100
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Sample input tokens: {input_ids[0, :10].tolist()}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    
    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Loss: {outputs.loss.item():.4f}")
    
    # Show predictions for first few tokens
    predictions = torch.argmax(outputs.logits[0, :5], dim=-1)
    print(f"Predictions for first 5 tokens: {predictions.tolist()}")
    
    return outputs


def demonstrate_chunking_behavior(model, config):
    """Demonstrate the HNet chunking behavior."""
    print("\nDemonstrating HNet chunking behavior...")
    
    # Create a longer sequence to see chunking effects
    batch_size = 1
    seq_len = 200
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Get hidden states to examine chunking
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
    
    print(f"Original sequence length: {seq_len}")
    print(f"Output sequence length: {outputs.logits.shape[1]}")
    
    # The model should maintain the same sequence length due to upsampling
    assert outputs.logits.shape[1] == seq_len, "Sequence length should be preserved"
    print("✓ Sequence length preserved through chunking and upsampling")


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("HNet-NTV2 Hybrid Model Demonstration")
    print("=" * 60)
    
    # Create model with NTV2 weights
    model, config = create_model_with_ntv2_weights()
    
    # Demonstrate forward pass
    outputs = demonstrate_forward_pass(model, config)
    
    # Demonstrate chunking behavior
    demonstrate_chunking_behavior(model, config)
    
    print("\n" + "=" * 60)
    print("Key Features:")
    print("✓ NTV2 transformer weights loaded into main layers")
    print("✓ HNet-style dynamic chunking and upsampling")
    print("✓ Trainable tokenization components")
    print("✓ Rotary positional embeddings (NTV2 style)")
    print("✓ Bidirectional processing capabilities")
    print("=" * 60)
    
    return model, config


if __name__ == "__main__":
    model, config = main()
