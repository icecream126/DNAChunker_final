#!/usr/bin/env python3
"""
Script to initialize HNet-NTV2 model with pre-trained NTV2 weights.
This script can be used to pre-load weights before training or as a standalone utility.
"""

import argparse
import torch
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hnet_ntv2.modeling_hnet_ntv2 import HNetNTV2Config, HNetNTV2ForMaskedLM, load_ntv2_weights


def main():
    parser = argparse.ArgumentParser(description="Initialize HNet-NTV2 with NTV2 weights")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory to save the initialized model")
    parser.add_argument("--config", type=str, 
                       default="configs/model/hnet_ntv2.yaml",
                       help="Path to model configuration file")
    parser.add_argument("--ntv2-model", type=str,
                       default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
                       help="NTV2 model name or path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for model initialization")
    parser.add_argument("--save-model", action="store_true",
                       help="Save the initialized model to disk")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("HNet-NTV2 Weight Initialization")
    print("=" * 50)
    print(f"Output directory: {args.output_dir}")
    print(f"Config file: {args.config}")
    print(f"NTV2 model: {args.ntv2_model}")
    print(f"Device: {args.device}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    print("Loading configuration...")
    config = HNetNTV2Config(
        d_model=1024,
        n_layer=24,
        vocab_size=4105,
        n_head=16,
        intermediate_size=8192,
        max_position_embeddings=1000,
        n_enc_layer=2,
        n_main_layer=20,
        n_dec_layer=2,
        target_ratio_stage1=0.3,
        target_ratio_stage2=0.5,
        tokenizer_type="default",
        bidirectional=True,
        bidirectional_strategy="add",
        bidirectional_weight_tie=True,
    )
    
    print(f"Configuration loaded: {config}")
    
    # Create model
    print("Creating HNet-NTV2 model...")
    model = HNetNTV2ForMaskedLM(config)
    model = model.to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load NTV2 weights
    print("Loading NTV2 weights into main transformer layers...")
    try:
        model = load_ntv2_weights(model, args.ntv2_model)
        print("NTV2 weights loaded successfully!")
    except Exception as e:
        print(f"Error loading NTV2 weights: {e}")
        print("Continuing with randomly initialized weights...")
    
    # Save model if requested
    if args.save_model:
        print("Saving initialized model...")
        model_path = os.path.join(args.output_dir, "hnet_ntv2_initialized.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'ntv2_model': args.ntv2_model,
        }, model_path)
        print(f"Model saved to: {model_path}")
        
        # Also save just the config
        config_path = os.path.join(args.output_dir, "config.json")
        config.save_pretrained(args.output_dir)
        print(f"Config saved to: {config_path}")
    
    print("=" * 50)
    print("Weight initialization completed!")
    print("=" * 50)
    
    return model, config


if __name__ == "__main__":
    model, config = main()

