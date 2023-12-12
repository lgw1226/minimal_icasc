#!/bin/bash

python test.py \
    --wandb-key 0f5cd9050587f427bc738060f38f870174f2c8e4 \
    --wandb-user hphp \
    --wandb-mode online \
    --exp-name Test-TIN-Class-1 \
    --parallel-block-channels 1 \
    --model-path trained_models/TIN-Class-1.pt \