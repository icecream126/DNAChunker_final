#!/bin/bash

# Set of dataset names to iterate through for genomics benchmark
DATASET_NAMES=(
    "dummy_mouse_enhancers_ensembl"
    "demo_coding_vs_intergenomic_seqs"
    "demo_human_or_worm"
    "human_enhancers_cohn"
    "human_enhancers_ensembl"
    "human_ensembl_regulatory"
    "human_nontata_promoters"
    "human_ocr_ensembl"
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
MAX_PROCESSES_PER_GPU=2
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
    local seed=$3
    local lr=$4
    local batch_size=$5
    local rc_aug=$6
    local pooling=$7
    
    echo "Starting training for dataset: $dataset_name on GPU: $gpu_id"
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    
    # Configuration paths for hnet model
    CFG_PATH="/workspace/outputs_inocras/pretrain/hg38/caduceus-hnet_seqlen-k_d_model-256_n_enc_layer-2_n_main_layer-8_n_dec_layer-2_lr-8e-3_tokenizer_type-default/model_config.json"
    CKPT_PATH="/workspace/outputs_inocras/pretrain/hg38/caduceus-hnet_seqlen-k_d_model-256_n_enc_layer-2_n_main_layer-8_n_dec_layer-2_lr-8e-3_tokenizer_type-default/checkpoints/test/loss.ckpt"
    
    # Set model configuration
    MODEL="caduceus_hnet"
    MODEL_NAME="dna_embedding_caduceus_hnet"
    CONJOIN_TRAIN_DECODER="false"
    CONJOIN_TEST="false"
    
    # Create wandb name and run directory
    WANDB_NAME="FULLTRAIN_USE_DECODER_HNET_${pooling}_${dataset_name}_lr-${lr}_batch_size-${batch_size}_rc_aug-${rc_aug}"
    HYDRA_RUN_DIR="./outputs/downstream/gb_cv5/${dataset_name}/${WANDB_NAME}/seed-${seed}"
    mkdir -p "${HYDRA_RUN_DIR}"
    
    echo "*****************************************************"
    echo "Running GenomicsBenchmark model: HNET, task: ${dataset_name}, lr: ${lr}, batch_size: ${batch_size}, rc_aug: ${rc_aug}, pooling: ${pooling}, SEED: ${seed}"
    
    # Run the training command
    python -m train \
        experiment=hg38/genomic_benchmark \
        callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
        dataset.dataset_name="${dataset_name}" \
        dataset.train_val_split_seed=${seed} \
        dataset.batch_size=${batch_size} \
        dataset.rc_aug="${rc_aug}" \
        +dataset.conjoin_train=false \
        +dataset.conjoin_test="${CONJOIN_TEST}" \
        model="${MODEL}" \
        model._name_="${MODEL_NAME}" \
        +model.config_path="${CFG_PATH}" \
        +model.conjoin_test="${CONJOIN_TEST}" \
        +decoder.conjoin_train="${CONJOIN_TRAIN_DECODER}" \
        +decoder.conjoin_test="${CONJOIN_TEST}" \
        optimizer.lr="${lr}" \
        trainer.max_epochs=10 \
        train.pretrained_model_path="${CKPT_PATH}" \
        wandb.group="downstream/gb_cv5_hnet" \
        wandb.job_type="${dataset_name}" \
        wandb.name="${WANDB_NAME}" \
        wandb.project="genomics_benchmark" \
        hydra.run.dir="${HYDRA_RUN_DIR}" \
        decoder.mode="${pooling}" 
    
    echo "Completed training for dataset: $dataset_name on GPU: $gpu_id"
    echo "*****************************************************"
    
    # Clean up run directory
    rm -rf "${HYDRA_RUN_DIR}"
}

# Main execution
echo "Starting genomics benchmark runs for ${#DATASET_NAMES[@]} datasets with HNET model"

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
        for seed in $(seq 2 3); do
            for lr in 1e-4 5e-4 1e-3 5e-3; do
                for batch_size in 64; do
    # for pooling in "len_pool"; do
    #     for seed in $(seq 1 1); do
    #         for lr in 5e-5; do
    #             for batch_size in 32; do
                    for rc_aug in "false"; do
                        echo "Waiting for available GPU for dataset: $dataset_name"
                        
                        # Wait for available GPU
                        while true; do
                            manage_gpus
                            
                            # Find available GPU
                            available_gpu=$(find_available_gpu)
                            
                            if [ -n "$available_gpu" ]; then
                                echo "Found available GPU: $available_gpu for dataset: $dataset_name"
                                
                                # Run training in background
                                run_training "$dataset_name" "$available_gpu" "$seed" "$lr" "$batch_size" "$rc_aug" "$pooling" &
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

echo "All genomics benchmark runs with HNET model completed!"
