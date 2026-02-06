@echo off
chcp 65001 > nul
setlocal

:: WANDB SETUP
set "WANDB_API_KEY=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"
set "WANDB_PROJECT=BSC Thesis"
set "WANDB_WATCH=false"
set "ADAPTER_ROOT=results\adapters"
set "DATASET=reuters"

:: We focus on REUTERS first to find the best recipe quickly.

echo ========================================================
echo   STARTING 4-WAY HYPERPARAMETER BATTLE
echo ========================================================

:: ----------------------------------------------------------
:: EXP 1: DoRA + Cosine Scheduler (The "Modern" Approach)
:: ----------------------------------------------------------
echo.
echo [1/4] Running Experiment A: DoRA + Cosine...
python src/train_lora.py --model e5_small --dataset %DATASET% --epochs 4 --batch_size 16 ^
  --use_dora --lr_scheduler cosine --suffix _expA

:: Rename folder
if exist "%ADAPTER_ROOT%\e5_small_%DATASET%" (
    if exist "%ADAPTER_ROOT%\e5_small_%DATASET%_expA" rmdir /S /Q "%ADAPTER_ROOT%\e5_small_%DATASET%_expA"
    move /Y "%ADAPTER_ROOT%\e5_small_%DATASET%" "%ADAPTER_ROOT%\e5_small_%DATASET%_expA"
)

:: Run Eval immediately to see result
echo [1/4] Evaluating Exp A...
python main.py --model e5_small --dataset %DATASET% --lora --suffix _expA --epochs 4
echo [1/4] DONE.


:: ----------------------------------------------------------
:: EXP 2: High Dropout + Low Rank (The "Anti-Overfit" Approach)
:: ----------------------------------------------------------
echo.
echo [2/4] Running Experiment B: High Dropout (0.3) + Rank 16...
python src/train_lora.py --model e5_small --dataset %DATASET% --epochs 4 --batch_size 16 ^
  --lora_dropout 0.3 --r 16 --lora_alpha 32 --suffix _expB

:: Rename folder
if exist "%ADAPTER_ROOT%\e5_small_%DATASET%" (
    if exist "%ADAPTER_ROOT%\e5_small_%DATASET%_expB" rmdir /S /Q "%ADAPTER_ROOT%\e5_small_%DATASET%_expB"
    move /Y "%ADAPTER_ROOT%\e5_small_%DATASET%" "%ADAPTER_ROOT%\e5_small_%DATASET%_expB"
)

:: Run Eval
echo [2/4] Evaluating Exp B...
python main.py --model e5_small --dataset %DATASET% --lora --suffix _expB --epochs 4
echo [2/4] DONE.


:: ----------------------------------------------------------
:: EXP 3: Bias Tuning (The "Subtle Style" Approach)
:: ----------------------------------------------------------
echo.
echo [3/4] Running Experiment C: Bias Tuning (lora_only)...
python src/train_lora.py --model e5_small --dataset %DATASET% --epochs 4 --batch_size 16 ^
  --bias lora_only --suffix _expC

:: Rename folder
if exist "%ADAPTER_ROOT%\e5_small_%DATASET%" (
    if exist "%ADAPTER_ROOT%\e5_small_%DATASET%_expC" rmdir /S /Q "%ADAPTER_ROOT%\e5_small_%DATASET%_expC"
    move /Y "%ADAPTER_ROOT%\e5_small_%DATASET%" "%ADAPTER_ROOT%\e5_small_%DATASET%_expC"
)

:: Run Eval
echo [3/4] Evaluating Exp C...
python main.py --model e5_small --dataset %DATASET% --lora --suffix _expC --epochs 4
echo [3/4] DONE.


:: ----------------------------------------------------------
:: EXP 4: "The Kitchen Sink" (Combine A + B + C)
:: ----------------------------------------------------------
echo.
echo [4/4] Running Experiment D: DoRA + Dropout 0.3 + Bias + Cosine...
python src/train_lora.py --model e5_small --dataset %DATASET% --epochs 4 --batch_size 16 ^
  --use_dora --lora_dropout 0.3 --bias lora_only --lr_scheduler cosine --suffix _expD

:: Rename folder
if exist "%ADAPTER_ROOT%\e5_small_%DATASET%" (
    if exist "%ADAPTER_ROOT%\e5_small_%DATASET%_expD" rmdir /S /Q "%ADAPTER_ROOT%\e5_small_%DATASET%_expD"
    move /Y "%ADAPTER_ROOT%\e5_small_%DATASET%" "%ADAPTER_ROOT%\e5_small_%DATASET%_expD"
)

:: Run Eval
echo [4/4] Evaluating Exp D...
python main.py --model e5_small --dataset %DATASET% --lora --suffix _expD --epochs 4
echo [4/4] DONE.

echo.
echo ========================================================
echo   ALL EXPERIMENTS COMPLETE. CHECK WANDB FOR THE WINNER.
echo ========================================================
pause