@echo off
setlocal

set EPOCHS=30
set SEEDS=42 0 1
set AUG_TYPES=none basic autoaugment randaugment augmix

for %%s in (%SEEDS%) do (
    for %%a in (%AUG_TYPES%) do (
        echo ========================================
        echo aug_type=%%a  seed=%%s  epochs=%EPOCHS%
        echo ========================================
        python scripts/train.py --aug_type %%a --seed %%s --epochs %EPOCHS%
        if errorlevel 1 (
            echo ERROR: aug_type=%%a seed=%%s failed.
            exit /b 1
        )
    )
)

echo All experiments done.
endlocal
