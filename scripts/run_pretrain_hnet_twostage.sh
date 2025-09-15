export HYDRA_FULL_ERROR=1

NUM_DEVICES=8

# Run script - MANUALLY SET THESE TWO VALUES
SEQLEN=2048
BATCH_SIZE=8
D_MODEL=1024
ENC_LAYER=4
MAIN_LAYER=8
DEC_LAYER=2
LR="5e-5"
TOKENIZER_TYPE="default"
TARGET_RATIO_STAGE1="0.3"
TARGET_RATIO_STAGE2="0.3"

# Calculate max steps and gradient accumulation to achieve target token count
# Target: 9,897,508,864 tokens
# Formula: batch_size * seqlen * max_steps * num_devices * accumulate_grad_batches = constant
TARGET_TOKENS=104855502848

# Calculate tokens per step
TOKENS_PER_STEP=$(( SEQLEN * BATCH_SIZE * NUM_DEVICES ))

# Calculate max steps to reach target token count
MAX_STEPS=$(( TARGET_TOKENS / TOKENS_PER_STEP ))

# Set gradient accumulation to 1 initially
ACCUMULATE_GRAD_BATCHES=1

# Ensure minimum values for stability
if [ $MAX_STEPS -lt 1000 ]; then
    MAX_STEPS=1000
fi

# If we need more effective batch size, increase gradient accumulation
# Target effective batch size similar to original Caduceus (128 * 4 = 512)
TARGET_EFFECTIVE_BATCH_SIZE=512
CURRENT_EFFECTIVE_BATCH_SIZE=$(( BATCH_SIZE * NUM_DEVICES * ACCUMULATE_GRAD_BATCHES ))

if [ $CURRENT_EFFECTIVE_BATCH_SIZE -lt $TARGET_EFFECTIVE_BATCH_SIZE ]; then
    ACCUMULATE_GRAD_BATCHES=$(( TARGET_EFFECTIVE_BATCH_SIZE / (BATCH_SIZE * NUM_DEVICES) ))
    # Recalculate max steps with new gradient accumulation
    MAX_STEPS=$(( TARGET_TOKENS / (TOKENS_PER_STEP * ACCUMULATE_GRAD_BATCHES) ))
fi

# Debug output
echo "Configuration:"
echo "  Sequence length: ${SEQLEN}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Tokens per step: ${TOKENS_PER_STEP}"
echo "  Max steps: ${MAX_STEPS}"
echo "  Accumulate grad batches: ${ACCUMULATE_GRAD_BATCHES}"
echo "  Effective batch size: $(( BATCH_SIZE * NUM_DEVICES * ACCUMULATE_GRAD_BATCHES ))"
echo "  Total tokens: $(( TOKENS_PER_STEP * MAX_STEPS * ACCUMULATE_GRAD_BATCHES ))"

# Calculate val_check_interval safely
VAL_CHECK_INTERVAL=$(( MAX_STEPS / 5 ))
if [ $VAL_CHECK_INTERVAL -gt $MAX_STEPS ]; then
    VAL_CHECK_INTERVAL=$MAX_STEPS
fi

# Ensure val_check_interval doesn't exceed reasonable limits
# PyTorch Lightning requires val_check_interval <= number of training batches
# Set a reasonable maximum of 1000 steps for validation
if [ $VAL_CHECK_INTERVAL -gt 1000 ]; then
    VAL_CHECK_INTERVAL=1000
fi

echo "  Val check interval: ${VAL_CHECK_INTERVAL}"

SEQLEN_DIS="$(echo "scale=0; ${SEQLEN} / 1000" | bc)k"
WANDB_NAME="WEIGHTED_hnet_twostage_seqlen-${SEQLEN_DIS}_d_model-${D_MODEL}_n_enc_layer-${ENC_LAYER}_n_main_layer-${MAIN_LAYER}_n_dec_layer-${DEC_LAYER}_lr-${LR}_tokenizer_type-${TOKENIZER_TYPE}"
HYDRA_RUN_DIR="./outputs/pretrain/hg38/${WANDB_NAME}"

mkdir -p "${HYDRA_RUN_DIR}"
python -m train \
  experiment=hg38/hnet_twostage \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  dataset.max_length=${SEQLEN} \
  dataset.batch_size=${BATCH_SIZE} \
  dataset.mlm=true \
  dataset.mlm_probability=0.15 \
  +dataset.motif_boundaries=false \
  model="hnet_twostage" \
  model.config.d_model=${D_MODEL} \
  model.config.n_enc_layer=${ENC_LAYER} \
  model.config.n_main_layer=${MAIN_LAYER} \
  model.config.n_dec_layer=${DEC_LAYER} \
  model.config.n_layer=$(( ENC_LAYER + MAIN_LAYER + DEC_LAYER )) \
  model.config.tokenizer_type=${TOKENIZER_TYPE} \
  model.config.target_ratio_stage1=${TARGET_RATIO_STAGE1} \
  model.config.target_ratio_stage2=${TARGET_RATIO_STAGE2} \
  optimizer.lr="${LR}" \
  train.global_batch_size=$(( BATCH_SIZE * NUM_DEVICES )) \
  trainer.max_steps=${MAX_STEPS} \
  trainer.devices=${NUM_DEVICES} \
  trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES} \
  trainer.gradient_clip_val=1 \
  +trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
  +trainer.strategy._target_=pytorch_lightning.strategies.DDPStrategy \
  +trainer.strategy.find_unused_parameters=true \
  wandb.group=pretrain_hg38 \
  wandb.name="${WANDB_NAME}" \
  dataset.repeat_penalty=0.1 \
  dataset.repeat_bed_file="/workspace/caduceus_proj/data/hg38/hg38_repeats.bed" \
  hydra.run.dir="${HYDRA_RUN_DIR}"
