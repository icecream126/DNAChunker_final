export HYDRA_FULL_ERROR=1

NUM_DEVICES=4

# Run script
# SEQLEN=131072
SEQLEN=131072
MAX_STEPS=50000
D_MODEL=256
N_ENC_LAYER=2
N_MAIN_LAYER=8
N_DEC_LAYER=2
LR="8e-3"
BIDIRECTIONAL_STRATEGY="add"
BIDIRECTIONAL_WEIGHT_TIE="true"
RC_AUG="false"
TARGET_RATIO="0.3"
TOKENIZER_TYPE="default"
# CKPT_PATH="/workspace/caduceus/outputs/pretrain/hg38/caduceus-hnet_seqlen-k_d_model-256_n_enc_layer-2_n_main_layer-8_n_dec_layer-2_lr-8e-3_tokenizer_type-default/checkpoints/test/loss.ckpt"

BATCH_SIZE=$(( 524288 / SEQLEN ))
# BATCH_SIZE=2
SEQLEN_DIS="$(echo "scale=0; ${SEQLEN} / 1000" | bc)k"
WANDB_NAME="caduceus-hnet_seqlen-${SEQLEN_DIS}_d_model-${D_MODEL}_n_enc_layer-${N_ENC_LAYER}_n_main_layer-${N_MAIN_LAYER}_n_dec_layer-${N_DEC_LAYER}_lr-${LR}_tokenizer_type-${TOKENIZER_TYPE}-motif"
HYDRA_RUN_DIR="./outputs/pretrain/hg38/${WANDB_NAME}"

echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "SEQLEN_DIS: ${SEQLEN_DIS}"
echo "WANDB_NAME: ${WANDB_NAME}"
echo "HYDRA_RUN_DIR: ${HYDRA_RUN_DIR}"

mkdir -p "${HYDRA_RUN_DIR}"
python -m train \
  experiment=hg38/caduceus_hnet \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  dataset.max_length=${SEQLEN} \
  dataset.batch_size=${BATCH_SIZE} \
  dataset.mlm=true \
  dataset.mlm_probability=0.15 \
  dataset.rc_aug="${RC_AUG}" \
  +dataset.motif_boundaries=true \
  model="caduceus_hnet" \
  model.config.d_model=${D_MODEL} \
  model.config.n_enc_layer=${N_ENC_LAYER} \
  model.config.n_main_layer=${N_MAIN_LAYER} \
  model.config.n_dec_layer=${N_DEC_LAYER} \
  model.config.bidirectional=true \
  model.config.bidirectional_strategy=${BIDIRECTIONAL_STRATEGY} \
  model.config.bidirectional_weight_tie=${BIDIRECTIONAL_WEIGHT_TIE} \
  model.config.target_ratio=${TARGET_RATIO} \
  model.config.tokenizer_type=${TOKENIZER_TYPE} \
  optimizer.lr="${LR}" \
  train.global_batch_size=$(( BATCH_SIZE * NUM_DEVICES )) \
  trainer.max_steps=${MAX_STEPS} \
  trainer.devices=${NUM_DEVICES} \
  +trainer.val_check_interval=$(( MAX_STEPS / 5 )) \
  trainer.accumulate_grad_batches=1 \
  wandb.group=pretrain_hg38 \
  wandb.name="${WANDB_NAME}" \
  hydra.run.dir="${HYDRA_RUN_DIR}"
