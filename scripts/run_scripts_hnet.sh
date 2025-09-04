#!/bin/bash

python -m train \
  experiment=hg38/hnet \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  dataset.max_length=1024 \
  dataset.batch_size=4 \
  dataset.mlm=true \
  dataset.mlm_probability=0.15 \
  dataset.rc_aug=false \
  optimizer.lr="8e-4" \
  train.global_batch_size=4 \
  trainer.max_steps=10000 \
  +trainer.val_check_interval=1000 \
  wandb=null \
  trainer.devices=1 \
  +train.aux_loss_weights.ratio_loss=0.03 \
  +train.aux_loss_weights.motif_loss=0.02
