#!/bin/bash

python main.py \
    --wandb-key 0f5cd9050587f427bc738060f38f870174f2c8e4 \
    --wandb-user hphp \
    --wandb-mode online \
    --exp-name TIN-Class-Att \
    --parallel-block-channels 1 \
    --use-att-loss \