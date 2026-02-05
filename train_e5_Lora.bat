@echo off
chcp 65001 > nul


:: ==========================================
:: 1. WANDB SETUP (REQUIRED)
:: ==========================================
set "WANDB_API_KEY=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
set "WANDB_PROJECT=BSC Thesis"
set "WANDB_WATCH=false"


echo ========================================
echo   STARTING 3-EPOCH TRAINING (E5 SMALL)
echo ========================================

:: Define where adapters are saved
set "ADAPTER_ROOT=results\adapters"

:: ------------------------------------------
:: 1. REUTERS (Train -> Rename)
:: ------------------------------------------
echo.
echo [1/2] Training REUTERS (3 Epochs)...
python src/train_lora.py --model e5_small --dataset reuters --epochs 3 --batch_size 16

echo    Renaming to e5_small_reuters_3ep...
if exist "%ADAPTER_ROOT%\e5_small_reuters" (
    move /Y "%ADAPTER_ROOT%\e5_small_reuters" "%ADAPTER_ROOT%\e5_small_reuters_3ep"
)

:: ------------------------------------------
:: 2. DARKREDDIT (Train -> Rename)
:: ------------------------------------------
echo.
echo [2/2] Training DARKREDDIT (3 Epochs)...
python src/train_lora.py --model e5_small --dataset darkreddit --epochs 3 --batch_size 16

echo    Renaming to e5_small_darkreddit_3ep...
if exist "%ADAPTER_ROOT%\e5_small_darkreddit" (
    move /Y "%ADAPTER_ROOT%\e5_small_darkreddit" "%ADAPTER_ROOT%\e5_small_darkreddit_3ep"
)

echo.
echo ========================================
echo   DONE. Adapters are named *_3ep
echo ========================================
pause