@echo off
chcp 65001 > nul
setlocal

set "WANDB_API_KEY=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"
set "WANDB_PROJECT=BSC Thesis"
set "ADAPTER_ROOT=results\adapters"

echo ========================================================
echo   STARTING LINEAR MARATHON (Clean Mode)
echo ========================================================


:: ========================================================
:: DARKREDDIT - EPOCH 2
:: ========================================================

echo.
echo [DarkReddit] Exp A (DoRA) - 2 Epochs...
python src/train_lora.py --model e5_small --dataset darkreddit --epochs 2 --batch_size 16 --use_dora --lr_scheduler cosine
move /Y "%ADAPTER_ROOT%\e5_small_darkreddit_2ep" "%ADAPTER_ROOT%\e5_small_darkreddit_expA_2ep" > nul

echo.
echo [DarkReddit] Exp B (Dropout) - 2 Epochs...
python src/train_lora.py --model e5_small --dataset darkreddit --epochs 2 --batch_size 16 --lora_dropout 0.3 --r 16 --lora_alpha 32
move /Y "%ADAPTER_ROOT%\e5_small_darkreddit_2ep" "%ADAPTER_ROOT%\e5_small_darkreddit_expB_2ep" > nul


:: ========================================================
:: DARKREDDIT - EPOCH 3
:: ========================================================

echo.
echo [DarkReddit] Exp A (DoRA) - 3 Epochs...
python src/train_lora.py --model e5_small --dataset darkreddit --epochs 3 --batch_size 16 --use_dora --lr_scheduler cosine
move /Y "%ADAPTER_ROOT%\e5_small_darkreddit_3ep" "%ADAPTER_ROOT%\e5_small_darkreddit_expA_3ep" > nul

echo.
echo [DarkReddit] Exp B (Dropout) - 3 Epochs...
python src/train_lora.py --model e5_small --dataset darkreddit --epochs 3 --batch_size 16 --lora_dropout 0.3 --r 16 --lora_alpha 32
move /Y "%ADAPTER_ROOT%\e5_small_darkreddit_3ep" "%ADAPTER_ROOT%\e5_small_darkreddit_expB_3ep" > nul


:: ========================================================
:: DARKREDDIT - EPOCH 4
:: ========================================================

echo.
echo [DarkReddit] Exp A (DoRA) - 4 Epochs...
python src/train_lora.py --model e5_small --dataset darkreddit --epochs 4 --batch_size 16 --use_dora --lr_scheduler cosine
move /Y "%ADAPTER_ROOT%\e5_small_darkreddit_4ep" "%ADAPTER_ROOT%\e5_small_darkreddit_expA_4ep" > nul

echo.
echo [DarkReddit] Exp B (Dropout) - 4 Epochs...
python src/train_lora.py --model e5_small --dataset darkreddit --epochs 4 --batch_size 16 --lora_dropout 0.3 --r 16 --lora_alpha 32
move /Y "%ADAPTER_ROOT%\e5_small_darkreddit_4ep" "%ADAPTER_ROOT%\e5_small_darkreddit_expB_4ep" > nul


:: ========================================================
:: DARKREDDIT - EPOCH 5
:: ========================================================

echo.
echo [DarkReddit] Exp A (DoRA) - 5 Epochs...
python src/train_lora.py --model e5_small --dataset darkreddit --epochs 5 --batch_size 16 --use_dora --lr_scheduler cosine
move /Y "%ADAPTER_ROOT%\e5_small_darkreddit_5ep" "%ADAPTER_ROOT%\e5_small_darkreddit_expA_5ep" > nul

echo.
echo [DarkReddit] Exp B (Dropout) - 5 Epochs...
python src/train_lora.py --model e5_small --dataset darkreddit --epochs 5 --batch_size 16 --lora_dropout 0.3