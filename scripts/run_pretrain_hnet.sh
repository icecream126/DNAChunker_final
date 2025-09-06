export HYDRA_FULL_ERROR=1

NUM_DEVICES=8

# Run script
SEQLEN=2048
MAX_STEPS=50000
D_MODEL=1024
ENC_LAYER=2
MAIN_LAYER=8
DEC_LAYER=2
LR="5e-4"
TOKENIZER_TYPE="default"
TARGET_RATIO="0.3"

BATCH_SIZE=8
SEQLEN_DIS="$(echo "scale=0; ${SEQLEN} / 1000" | bc)k"
WANDB_NAME="hnet_seqlen-${SEQLEN_DIS}_d_model-${D_MODEL}_n_enc_layer-${ENC_LAYER}_n_main_layer-${MAIN_LAYER}_n_dec_layer-${DEC_LAYER}_lr-${LR}_tokenizer_type-${TOKENIZER_TYPE}"
HYDRA_RUN_DIR="./outputs/pretrain/hg38/${WANDB_NAME}"

mkdir -p "${HYDRA_RUN_DIR}"
python -m train \
  experiment=hg38/hnet \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  dataset.max_length=${SEQLEN} \
  dataset.batch_size=${BATCH_SIZE} \
  dataset.mlm=true \
  dataset.mlm_probability=0.15 \
  +dataset.motif_boundaries=false \
  loader.num_workers=0 \
  model="hnet" \
  model.config.d_model=${D_MODEL} \
  model.config.n_enc_layer=${ENC_LAYER} \
  model.config.n_main_layer=${MAIN_LAYER} \
  model.config.n_dec_layer=${DEC_LAYER} \
  model.config.n_layer=$(( ENC_LAYER + MAIN_LAYER + DEC_LAYER )) \
  model.config.tokenizer_type=${TOKENIZER_TYPE} \
  model.config.target_ratio=${TARGET_RATIO} \
  optimizer.lr="${LR}" \
  train.global_batch_size=$(( BATCH_SIZE * NUM_DEVICES )) \
  trainer.max_steps=${MAX_STEPS} \
  trainer.devices=${NUM_DEVICES} \
  trainer.accumulate_grad_batches=1 \
  trainer.gradient_clip_val=1 \
  +trainer.strategy=ddp \
  +trainer.val_check_interval=1 \
  wandb.group=pretrain_hg38 \
  wandb.name="${WANDB_NAME}" \
  hydra.run.dir="${HYDRA_RUN_DIR}"
