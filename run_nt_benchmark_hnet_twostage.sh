#!/bin/bash

# Set of dataset names to iterate through
DATASET_NAMES=(
    # "enhancers"
    # "enhancers_types"
    # "H3"
    # "H3K4me1"
    # "H3K4me2"
    # "H3K4me3"
    # "H3K9ac"
    # "H3K14ac"
    # "H3K36me3"
    # "H3K79me3"
    "H4"
    # "H4ac"
    # "promoter_all"
    "promoter_no_tata"
    "promoter_tata"
    # "splice_sites_acceptors"
    "splice_sites_all"
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

# Maximum number of processes per GPU
MAX_PROCESSES_PER_GPU=1
echo "Maximum processes per GPU: $MAX_PROCESSES_PER_GPU"

# Function to find available GPU
find_available_gpu() {
    # Check each specified GPU for availability
    for gpu_id in "${AVAILABLE_GPUS[@]}"; do
        # Check current process count for this GPU
        current_processes=${gpu_process_counts[$gpu_id]:-0}
        
        # Skip if already at max processes
        if [ "$current_processes" -ge "$MAX_PROCESSES_PER_GPU" ]; then
            continue
        fi
        
        # Check GPU memory usage only
        gpu_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | grep "^$gpu_id,")
        
        if [ -n "$gpu_info" ]; then
            memory_used=$(echo "$gpu_info" | awk -F', ' '{print $2}')
            memory_total=$(echo "$gpu_info" | awk -F', ' '{print $3}')
            
            # Calculate 60% memory threshold
            memory_threshold=$(awk "BEGIN {printf \"%.0f\", $memory_total * 0.6}")
            
            # Check if GPU has capacity (memory usage below 60% and under process limit)
            if [ "$memory_used" -lt "$memory_threshold" ]; then
                echo $gpu_id
                return 0
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
    local lr=$4
    local pooling=$5
    echo "Starting training for dataset: $dataset_name on GPU: $gpu_id"
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=$gpu_id

    local WANDB_NAME="full_train_RECENT_HNET_TWOSTAGE_WEIGHTED_${pooling}_${dataset_name}_val_idx-${val_idx}_lr-${lr}"
    local HYDRA_RUN_DIR="./outputs/downstream/nt/${WANDB_NAME}"

    rm -rf "${HYDRA_RUN_DIR}"
    mkdir -p "${HYDRA_RUN_DIR}"

    CFG_PATH="/workspace/outputs_twostage/WEIGHTED_hnet_twostage_seqlen-k_d_model-1024_n_enc_layer-4_n_main_layer-8_n_dec_layer-2_lr-5e-5_tokenizer_type-default/model_config.json"
    CKPT_PATH="/workspace/outputs_twostage/WEIGHTED_hnet_twostage_seqlen-k_d_model-1024_n_enc_layer-4_n_main_layer-8_n_dec_layer-2_lr-5e-5_tokenizer_type-default/checkpoints/val/loss.ckpt"

    # Run the training command
    python -m train \
        experiment=hg38/nucleotide_transformer \
        callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
        dataset.train_val_split_seed=$val_idx \
        dataset.batch_size=32 \
        dataset.rc_aug=false \
        +dataset.conjoin_test=false \
        model._name_=dna_embedding_hnet_twostage \
        +model.config_path=${CFG_PATH} \
        +model.conjoin_test=false \
        +decoder.conjoin_train=false \
        +decoder.conjoin_test=false \
        optimizer.lr=${lr} \
        train.pretrained_model_path=${CKPT_PATH} \
        trainer.max_epochs=20 \
        dataset.dataset_name=$dataset_name \
        train.monitor=val/proper_mcc \
        wandb.name=${WANDB_NAME} \
        decoder.mode=${pooling} \
        trainer.precision=32 \
        +wandb.tags=\["seed-${seed}"\] \
        hydra.run.dir="${HYDRA_RUN_DIR}" \
        model=hnet_twostage 
    echo "Completed training and testing for dataset: $dataset_name on GPU: $gpu_id"
    rm -rf "${HYDRA_RUN_DIR}"
}

# Main execution
echo "Starting benchmark runs for ${#DATASET_NAMES[@]} datasets"

# Track running processes - each GPU can have multiple processes
declare -A running_processes  # Maps PID to GPU_ID
declare -A gpu_process_counts # Maps GPU_ID to number of running processes

# Function to check and manage GPU assignments
manage_gpus() {
    # Check all running processes
    for pid in "${!running_processes[@]}"; do
        if ! kill -0 $pid 2>/dev/null; then
            # Process completed, update GPU count
            gpu_id=${running_processes[$pid]}
            echo "Process $pid on GPU $gpu_id completed"
            
            # Decrease process count for this GPU
            if [ -n "${gpu_process_counts[$gpu_id]}" ]; then
                ((gpu_process_counts[$gpu_id]--))
                if [ "${gpu_process_counts[$gpu_id]}" -le 0 ]; then
                    unset gpu_process_counts[$gpu_id]
                fi
            fi
            
            # Remove process from tracking
            unset running_processes[$pid]
        fi
    done
}

# Run training for each dataset
for dataset_name in "${DATASET_NAMES[@]}"; do
    for pooling in "pool"; do
        # for val_idx in $(seq 6 8); do
        #     for lr in 6e-5 7e-5 8e-5 9e-5; do
        for val_idx in $(seq 6 8); do
            for lr in 5e-5 8e-5; do
                echo "Waiting for available GPU for dataset: $dataset_name"
                
                # Wait for available GPU
                while true; do
                    manage_gpus
                    
                    # Find available GPU
                    available_gpu=$(find_available_gpu)
                    
                    if [ -n "$available_gpu" ]; then
                        echo "Found available GPU: $available_gpu for dataset: $dataset_name"
                        
                        # Run training in background
                        run_training "$dataset_name" "$available_gpu" "$val_idx" "$lr" "$pooling" &
                        process_pid=$!
                        
                        # Track the process
                        running_processes[$process_pid]=$available_gpu
                        
                        # Update GPU process count
                        if [ -z "${gpu_process_counts[$available_gpu]}" ]; then
                            gpu_process_counts[$available_gpu]=1
                        else
                            ((gpu_process_counts[$available_gpu]++))
                        fi
                        
                        echo "Started training for $dataset_name on GPU $available_gpu (PID: $process_pid, GPU processes: ${gpu_process_counts[$available_gpu]}/$MAX_PROCESSES_PER_GPU)"
                        break
                    fi
                    
                    echo "No GPU available, waiting..."
                    sleep 10
                done
            done
        done
    done
done

# Wait for all processes to complete
echo "Waiting for all training runs to complete..."
for pid in "${!running_processes[@]}"; do
    if [ -n "$pid" ]; then
        gpu_id=${running_processes[$pid]}
        echo "Waiting for process $pid on GPU $gpu_id"
        wait $pid
    fi
done

echo "All benchmark runs completed!"
