# HNet-NTV2 Hybrid Model

This package provides a hybrid model that combines the best of both worlds:
- **Nucleotide Transformer v2 (NTV2)** as the main transformer backbone with pre-trained weights
- **HNet-style tokenization and dynamic chunking** for efficient sequence processing

## Key Features

- **Pre-trained NTV2 weights**: The main transformer layers are initialized with weights from the 500M parameter NTV2 model
- **HNet tokenization**: Uses HNet-style trainable tokenization instead of NTV2's 6-mer tokenizer
- **Dynamic chunking**: Implements HNet's dynamic chunking mechanism for efficient long sequence processing
- **Rotary embeddings**: Maintains NTV2's rotary positional embeddings
- **Trainable components**: All tokenization and chunking components are trainable

## Architecture

The model consists of three main parts:

1. **Encoder layers** (2 layers): Process input sequences and prepare for chunking
2. **Main transformer layers** (20 layers): Core NTV2 transformer with loaded pre-trained weights
3. **Decoder layers** (2 layers): Process chunked representations and upsample back to original resolution

## Usage

### Basic Usage

```python
from modeling_hnet_ntv2 import HNetNTV2Config, HNetNTV2ForMaskedLM, load_ntv2_weights

# Create configuration
config = HNetNTV2Config(
    d_model=512,           # NTV2 hidden size
    n_layer=24,            # NTV2 number of layers
    vocab_size=4105,       # NTV2 vocabulary size
    n_head=8,              # NTV2 attention heads
    intermediate_size=2048, # NTV2 MLP intermediate size
    max_position_embeddings=1000,
    
    # HNet-specific configuration
    n_enc_layer=2,         # Encoder layers
    n_main_layer=20,       # Main transformer layers (NTV2 weights)
    n_dec_layer=2,         # Decoder layers
    tokenizer_type="hnet", # HNet-style tokenization
    target_ratio=0.25,     # Chunking ratio
)

# Create model
model = HNetNTV2ForMaskedLM(config)

# Load NTV2 weights into main transformer layers
model = load_ntv2_weights(model)
```

### Training

```python
import torch

# Create dummy data
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
labels = input_ids.clone()
# Mask some tokens for MLM
mask_indices = torch.rand(batch_size, seq_len) < 0.15
labels[~mask_indices] = -100

# Forward pass
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss

# Backward pass
loss.backward()
```

### Running Tests

```bash
cd hnet_ntv2
python test_hnet_ntv2.py
```

### Running Example

```bash
cd hnet_ntv2
python example_usage.py
```

## Model Components

### HNet Tokenization
- Trainable word embeddings
- Positional embeddings
- Layer normalization and dropout

### Dynamic Chunking
- Routing module for boundary prediction
- Chunk layer for downsampling
- Cross-attention upsampler for dechunking

### NTV2 Transformer
- Multi-head self-attention with rotary embeddings
- Gated Linear Units (GLU) in MLP
- Layer normalization
- Pre-trained weights from NTV2-500M

## Configuration Parameters

- `d_model`: Hidden dimension (512 for NTV2)
- `n_layer`: Total number of layers (24 for NTV2)
- `vocab_size`: Vocabulary size (4105 for NTV2)
- `n_head`: Number of attention heads (8 for NTV2)
- `intermediate_size`: MLP intermediate size (2048 for NTV2)
- `max_position_embeddings`: Maximum sequence length (1000 for NTV2)
- `n_enc_layer`: Number of encoder layers (2)
- `n_main_layer`: Number of main transformer layers (20)
- `n_dec_layer`: Number of decoder layers (2)
- `tokenizer_type`: Tokenization type ("hnet")
- `target_ratio`: Chunking ratio (0.25)
- `use_rotary_embeddings`: Use rotary positional embeddings (True)

## Dependencies

- PyTorch
- Transformers
- NTV2 model weights (downloaded automatically)

## Notes

- The model automatically downloads NTV2 weights when `load_ntv2_weights()` is called
- All HNet components (tokenization, chunking) are trainable
- The main transformer layers are initialized with NTV2 weights but can be fine-tuned
- Sequence length is preserved through the chunking and upsampling process
