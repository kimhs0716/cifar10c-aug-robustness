#!/usr/bin/env bash
set -e

EPOCHS=30
SEEDS=(42 0 1)
AUG_TYPES=(none basic autoaugment randaugment augmix)

for seed in "${SEEDS[@]}"; do
    for aug in "${AUG_TYPES[@]}"; do
        run_name="resnet18_${aug}_seed${seed}"
        if [ -f "results/${run_name}/metrics.json" ]; then
            echo "SKIP: $run_name (already done)"
            continue
        fi
        echo "========================================"
        echo "aug_type=$aug  seed=$seed  epochs=$EPOCHS"
        echo "========================================"
        python scripts/train.py --aug_type "$aug" --seed "$seed" --epochs "$EPOCHS" || {
            echo "ERROR: $run_name failed."
            exit 1
        }
    done
done

echo "All experiments done."
