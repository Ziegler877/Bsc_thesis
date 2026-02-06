@echo off
chcp 65001 > nul
setlocal

:: ========================================================
:: CONFIGURATION
:: ========================================================
set "WANDB_API_KEY=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"
set "WANDB_PROJECT=BSC Thesis"
set "WANDB_WATCH=false"
set "ADAPTER_ROOT=results\adapters"
set "DATASET=reuters"
set "EPOCHS=3"

echo ========================================================
echo   STARTING 4-WAY TRAINING BATTLE (3 Epochs)
echo   Script will ONLY TRAIN. No Evaluation.
echo ========================================================

:: ----------------------------------------------------------
:: EXP A: DoRA + Cosine Scheduler
:: ----------------------------------------------------------
echo.
echo [1/4] Training Experiment A: DoRA + Cosine...
python src/train_lora.py --model e5_small --dataset %DATASET% --epochs %EPOCHS% --batch_size 16 ^
  --use_dora --lr_scheduler cosine

:: Rename Folder to _expA
if exist "%ADAPTER_ROOT%\e5_small_%DATASET%" (
    echo    Renaming folder to ..._expA
    if exist "%ADAPTER_ROOT%\e5_small_%DATASET%_expA" rmdir /S /Q "%ADAPTER_ROOT%\e5_small_%DATASET%_expA"
    move /Y "%ADAPTER_ROOT%\e5_small_%DATASET%" "%ADAPTER_ROOT%\e5_small_%DATASET%_expA" > nul
) else (
    echo [ERROR] Training A failed. No folder found.
)
echo [1/4] DONE.


:: ----------------------------------------------------------
:: EXP B: High Dropout + Low Rank
:: ----------------------------------------------------------
echo.
echo [2/4] Training Experiment B: Dropout 0.3 + Rank 16...
python src/train_lora.py --model e5_small --dataset %DATASET% --epochs %EPOCHS% --batch_size 16 ^
  --lora_dropout 0.3 --r 16 --lora_alpha 32

:: Rename Folder to _expB
if exist "%ADAPTER_ROOT%\e5_small_%DATASET%" (
    echo    Renaming folder to ..._expB
    if exist "%ADAPTER_ROOT%\e5_small_%DATASET%_expB" rmdir /S /Q "%ADAPTER_ROOT%\e5_small_%DATASET%_expB"
    move /Y "%ADAPTER_ROOT%\e5_small_%DATASET%" "%ADAPTER_ROOT%\e5_small_%DATASET%_expB" > nul
) else (
    echo [ERROR] Training B failed. No folder found.
)
echo [2/4] DONE.


:: ----------------------------------------------------------
:: EXP C: Bias Tuning
:: ----------------------------------------------------------
echo.
echo [3/4] Training Experiment C: Bias Tuning (lora_only)...
python src/train_lora.py --model e5_small --dataset %DATASET% --epochs %EPOCHS% --batch_size 16 ^
  --bias lora_only

:: Rename Folder to _expC
if exist "%ADAPTER_ROOT%\e5_small_%DATASET%" (
    echo    Renaming folder to ..._expC
    if exist "%ADAPTER_ROOT%\e5_small_%DATASET%_expC" rmdir /S /Q "%ADAPTER_ROOT%\e5_small_%DATASET%_expC"
    move /Y "%ADAPTER_ROOT%\e5_small_%DATASET%" "%ADAPTER_ROOT%\e5_small_%DATASET%_expC" > nul
) else (
    echo [ERROR] Training C failed. No folder found.
)
echo [3/4] DONE.


:: ----------------------------------------------------------
:: EXP D: The Kitchen Sink (Combo)
:: ----------------------------------------------------------
echo.
echo [4/4] Training Experiment D: DoRA + Dropout 0.3 + Bias + Cosine...
python src/train_lora.py --model e5_small --dataset %DATASET% --epochs %EPOCHS% --batch_size 16 ^
  --use_dora --lora_dropout 0.3 --bias lora_only --lr_scheduler cosine

:: Rename Folder to _expD
if exist "%ADAPTER_ROOT%\e5_small_%DATASET%" (
    echo    Renaming folder to ..._expD
    if exist "%ADAPTER_ROOT%\e5_small_%DATASET%_expD" rmdir /S /Q "%ADAPTER_ROOT%\e5_small_%DATASET%_expD"
    move /Y "%ADAPTER_ROOT%\e5_small_%DATASET%" "%ADAPTER_ROOT%\e5_small_%DATASET%_expD" > nul
) else (
    echo [ERROR] Training D failed. No folder found.
)
echo [4/4] DONE.


echo.
echo ========================================================
echo   ALL TRAININGS COMPLETE.
echo   Check results/adapters for folders ending in expA-D
echo ========================================================
pause