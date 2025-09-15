# HNet-NTV2 Hybrid Model for hg38 Tasks

This directory contains scripts and configurations for running the HNet-NTV2 hybrid model on hg38 genomic tasks. The HNet-NTV2 model combines:

- **HNet-style tokenization and dynamic chunking** for efficient sequence processing
- **Pre-trained Nucleotide Transformer v2 (NTV2) weights** in the main transformer layers
- **Two-stage upsampling** for sequence reconstruction

## Model Architecture

The HNet-NTV2 model consists of:

1. **Encoder layers** (2 layers): Process input sequences with HNet tokenization
2. **Main transformer layers** (20 layers): Loaded with pre-trained NTV2 weights
3. **Decoder layers** (2 layers): Two-stage upsampling for sequence reconstruction

### Key Features

- **Pre-trained backbone**: Main transformer layers initialized with NTV2 weights
- **Dynamic chunking**: Efficient sequence compression using HNet's routing mechanism
- **Two-stage reconstruction**: Gradual upsampling for better sequence recovery
- **Bidirectional processing**: Handles both forward and reverse complement sequences

## Files

### Configuration Files

- `configs/model/hnet_ntv2.yaml`: Model configuration with NTV2-compatible parameters
- `configs/experiment/hg38/hnet_ntv2.yaml`: Experiment configuration for hg38 tasks

### Scripts

- `run_pretrain_hnet_ntv2.sh`: Main training script with command-line options
- `init_hnet_ntv2_weights.py`: Utility to initialize model with NTV2 weights

## Quick Start

### 1. Basic Training

```bash
# Run with default settings
./scripts/run_pretrain_hnet_ntv2.sh

# Run with custom settings
./scripts/run_pretrain_hnet_ntv2.sh \
    --experiment-name "my_hnet_ntv2" \
    --output-dir "outputs/my_experiment" \
    --max-steps 100000 \
    --batch-size 256 \
    --learning-rate 3e-4
```

### 2. Initialize Weights Only

```bash
# Initialize model with NTV2 weights and save
python scripts/init_hnet_ntv2_weights.py \
    --output-dir "outputs/initialized_model" \
    --save-model
```

### 3. Custom Configuration

```bash
# Use custom model configuration
./scripts/run_pretrain_hnet_ntv2.sh \
    --config-path "configs/experiment/hg38/my_custom_config.yaml"
```

## Configuration Options

### Model Parameters

- `d_model`: 1024 (NTV2 hidden size)
- `n_head`: 16 (NTV2 attention heads)
- `intermediate_size`: 8192 (NTV2 MLP intermediate size)
- `n_main_layer`: 20 (transformer layers with NTV2 weights)
- `n_enc_layer`: 2 (encoder layers)
- `n_dec_layer`: 2 (decoder layers)

### Training Parameters

- `max_steps`: 50000 (default training steps)
- `batch_size`: 512 (adjusted for larger model)
- `learning_rate`: 6e-4 (lower for fine-tuning)
- `precision`: 16 (mixed precision for efficiency)

### HNet Parameters

- `target_ratio_stage1`: 0.3 (first stage compression ratio)
- `target_ratio_stage2`: 0.5 (second stage compression ratio)
- `bidirectional`: true (process both strands)
- `tokenizer_type`: "default" (HNet tokenization)

## Command Line Options

The training script supports the following options:

```bash
--experiment-name NAME     # Name of the experiment
--config-path PATH         # Path to config file
--output-dir DIR           # Output directory
--wandb-project PROJECT    # Weights & Biases project
--wandb-entity ENTITY      # Weights & Biases entity
--devices N                # Number of devices
--num-nodes N              # Number of nodes
--max-steps N              # Maximum training steps
--batch-size N             # Batch size
--learning-rate LR         # Learning rate
--precision P              # Precision (16 or 32)
```

## Memory Requirements

The HNet-NTV2 model is larger than standard HNet models due to:

- **Larger hidden size**: 1024 (vs 128 in standard HNet)
- **More attention heads**: 16 (vs 8 in standard HNet)
- **Larger MLP**: 8192 intermediate size
- **More parameters**: ~432M total parameters

### Recommended Hardware

- **GPU Memory**: 24GB+ (RTX 3090, A100, etc.)
- **Batch Size**: 256-512 (adjust based on GPU memory)
- **Precision**: 16-bit mixed precision recommended

## Monitoring Training

The script supports Weights & Biases logging:

```bash
./scripts/run_pretrain_hnet_ntv2.sh \
    --wandb-project "hnet-ntv2-experiments" \
    --wandb-entity "your-username"
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **NTV2 Download Issues**: Ensure internet connection and sufficient disk space
3. **Weight Loading Errors**: Check that NTV2 model is accessible

### Debug Mode

Run with smaller parameters for testing:

```bash
./scripts/run_pretrain_hnet_ntv2.sh \
    --max-steps 1000 \
    --batch-size 64 \
    --devices 1
```

## Expected Performance

The HNet-NTV2 model should show:

- **Better initialization**: Pre-trained NTV2 weights provide good starting point
- **Faster convergence**: Reduced training time compared to random initialization
- **Better performance**: Improved accuracy on genomic tasks
- **Efficient processing**: HNet's dynamic chunking for long sequences

## File Structure

```
caduceus_proj/
├── configs/
│   ├── model/hnet_ntv2.yaml
│   └── experiment/hg38/hnet_ntv2.yaml
├── scripts/
│   ├── run_pretrain_hnet_ntv2.sh
│   ├── init_hnet_ntv2_weights.py
│   └── README_hnet_ntv2.md
├── hnet_ntv2/
│   ├── modeling_hnet_ntv2.py
│   ├── configuration_hnet_ntv2.py
│   └── __init__.py
└── outputs/
    └── hnet_ntv2/
        ├── checkpoints/
        ├── logs/
        └── configs/
```

## Next Steps

1. **Run initial training** with default settings
2. **Monitor performance** using Weights & Biases
3. **Tune hyperparameters** based on validation metrics
4. **Scale up training** with more steps or larger batch sizes
5. **Evaluate on downstream tasks** using the trained model

For questions or issues, please refer to the main project documentation or create an issue in the repository.
