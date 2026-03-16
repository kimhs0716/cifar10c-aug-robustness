#!/usr/bin/env bash
set -e

EPOCHS=30
SEEDS=(42 0 1)
AUG_TYPES=(none basic autoaugment randaugment augmix)

for seed in "${SEEDS[@]}"; do
    for aug in "${AUG_TYPES[@]}"; do
        echo "========================================"
        echo "aug_type=$aug  seed=$seed  epochs=$EPOCHS"
        echo "========================================"
        python scripts/train.py --aug_type "$aug" --seed "$seed" --epochs "$EPOCHS"
    done
done

echo "All experiments done."
