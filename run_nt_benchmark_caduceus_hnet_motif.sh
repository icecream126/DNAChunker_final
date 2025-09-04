#!/bin/bash

# Set of dataset names to iterate through
DATASET_NAMES=(
    "promoter_all"
    "promoter_no_tata"
    "promoter_tata"
    "enhancers"
    "enhancers_types"
    "H3"
    "H3K4me1"
    "H3K4me2"
    "H3K4me3"
    "H3K9ac"
    "splice_sites_acceptors"
    "splice_sites_donors"
)

# Parse command line arguments for GPU IDs
AVAILABLE_GPUS=()
if [ $# -eq 0 ]; then
    echo "Usage: $0 <gpu_id1> [gpu_id2] [gpu_id3] ..."
    echo "Example: $0 0 1 2"
    echo "If no GPUs specified, will auto-detect available ones"
    # Auto-detect all GPUs
    AVAILABLE_GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader,nounits))
else
    AVAILABLE_GPUS=("$@")
fi

echo "Using GPUs: ${AVAILABLE_GPUS[*]}"

# Function to find available GPU
find_available_gpu() {
    # Check each specified GPU for availability
    for gpu_id in "${AVAILABLE_GPUS[@]}"; do
        if [ -z "${gpu_assignments[$gpu_id]}" ]; then
            # Check if GPU memory usage is below threshold
            memory_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | grep "^$gpu_id,")
            if [ -n "$memory_info" ]; then
                memory_used=$(echo "$memory_info" | awk -F', ' '{print $2}')
                memory_total=$(echo "$memory_info" | awk -F', ' '{print $3}')
                if [ "$memory_used" -lt "$((memory_total * 80 / 100))" ]; then
                    echo $gpu_id
                    return 0
                fi
            fi
        fi
    done
    
    # No available GPU found
    echo ""
}

# Function to run training with a specific dataset and GPU
run_training() {
    local dataset_name=$1
    local gpu_id=$2
    local val_idx=$3
    
    echo "Starting training for dataset: $dataset_name on GPU: $gpu_id"
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Run the training command
    python -m train \
        experiment=hg38/nucleotide_transformer \
        callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
        dataset.train_val_split_seed=$val_idx \
        dataset.batch_size=2048 \
        dataset.rc_aug=false \
        +dataset.conjoin_test=false \
        model._name_=dna_embedding_caduceus_hnet_motif \
        +model.config_path=/workspace/caduceus/outputs/2025-08-31/caduceus_hnet_motif/model_config.json \
        +model.conjoin_test=false \
        +decoder.conjoin_train=false \
        +decoder.conjoin_test=false \
        optimizer.lr=5e-4 \
        train.pretrained_model_path=/workspace/caduceus/outputs/2025-08-31/caduceus_hnet_motif/checkpoints/last.ckpt \
        trainer.max_epochs=20 \
        dataset.dataset_name=$dataset_name \
        model=caduceus_hnet 
    echo "Completed training for dataset: $dataset_name on GPU: $gpu_id"
}

# Main execution
echo "Starting benchmark runs for ${#DATASET_NAMES[@]} datasets"

# Track running processes
declare -A running_processes
declare -A gpu_assignments

# Function to check and manage GPU assignments
manage_gpus() {
    for gpu_id in "${!gpu_assignments[@]}"; do
        if [ -n "${running_processes[$gpu_id]}" ]; then
            # Check if process is still running
            if ! kill -0 ${running_processes[$gpu_id]} 2>/dev/null; then
                echo "Process on GPU $gpu_id completed, freeing up GPU"
                unset running_processes[$gpu_id]
                unset gpu_assignments[$gpu_id]
            fi
        fi
    done
}

# Run training for each dataset
for dataset_name in "${DATASET_NAMES[@]}"; do
    for val_idx in $(seq 0 2); do
        echo "Waiting for available GPU for dataset: $dataset_name"
        
        # Wait for available GPU
        while true; do
            manage_gpus
            
            # Find available GPU
            available_gpu=$(find_available_gpu)
            
            if [ -n "$available_gpu" ] && [ -z "${gpu_assignments[$available_gpu]}" ]; then
                echo "Found available GPU: $available_gpu for dataset: $dataset_name"
                
                # Assign GPU and start training
                gpu_assignments[$available_gpu]=$dataset_name
                
                # Run training in background
                run_training "$dataset_name" "$available_gpu" "$val_idx" &
                running_processes[$available_gpu]=$!
                
                echo "Started training for $dataset_name on GPU $available_gpu (PID: ${running_processes[$available_gpu]})"
                break
            fi
            
            echo "No GPU available, waiting..."
            sleep 10
        done
    done
done

# Wait for all processes to complete
echo "Waiting for all training runs to complete..."
for gpu_id in "${!running_processes[@]}"; do
    if [ -n "${running_processes[$gpu_id]}" ]; then
        echo "Waiting for process on GPU $gpu_id (PID: ${running_processes[$gpu_id]})"
        wait ${running_processes[$gpu_id]}
    fi
done

echo "All benchmark runs completed!"
