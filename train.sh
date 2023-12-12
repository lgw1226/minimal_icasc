#!/bin/bash

python train.py \
    --wandb-key 0f5cd9050587f427bc738060f38f870174f2c8e4 \
    --wandb-user hphp \
    --wandb-mode online \
    --exp-name TIN-Class-Att-12 \
    --parallel-block-channels 12 \
    --use-att-loss \