#!/bin/bash

python train.py \
    --wandb-key 0f5cd9050587f427bc738060f38f870174f2c8e4 \
    --wandb-user hphp \
    --wandb-mode online \
    --exp-name TIN-Class-BW-14 \
    --parallel-block-channels 14 \
    --num-epochs 120 \
    --milestones 30 60 90 \
    --use-bw-loss \