export HYDRA_FULL_ERROR=1

NUM_DEVICES=1

# Run script
SEQLEN=131072
MAX_STEPS=50000
D_MODEL=256
N_LAYER=8
LR="5e-6"
TOKENIZER_TYPE="default"
TARGET_RATIO="0.3"

BATCH_SIZE=$(( 1048576 / SEQLEN ))
SEQLEN_DIS="$(echo "scale=0; ${SEQLEN} / 1000" | bc)k"
WANDB_NAME="hnet_seqlen-${SEQLEN_DIS}_d_model-${D_MODEL}_n_layer-${N_LAYER}_lr-${LR}_tokenizer_type-${TOKENIZER_TYPE}"
HYDRA_RUN_DIR="./outputs/pretrain/hg38/${WANDB_NAME}"

mkdir -p "${HYDRA_RUN_DIR}"
python -m train \
  experiment=hg38/hnet \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  dataset.max_length=${SEQLEN} \
  dataset.batch_size=2 \
  dataset.mlm=true \
  dataset.mlm_probability=0.15 \
  model="hnet" \
  model.config.d_model=${D_MODEL} \
  model.config.n_layer=${N_LAYER} \
  model.config.tokenizer_type=${TOKENIZER_TYPE} \
  model.config.target_ratio=${TARGET_RATIO} \
  optimizer.lr="${LR}" \
  train.global_batch_size=2 \
  trainer.max_steps=${MAX_STEPS} \
  trainer.devices=${NUM_DEVICES} \
  trainer.accumulate_grad_batches=8 \
  +trainer.val_check_interval=$(( MAX_STEPS / 5 )) \
  wandb.group=pretrain_hg38 \
  wandb.name="${WANDB_NAME}" \
  hydra.run.dir="${HYDRA_RUN_DIR}"
