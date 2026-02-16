@echo off
chcp 65001
setlocal

:: ====================================================
:: SETUP LOGGING
:: ====================================================

:: Generate a unique log filename
set "TIMESTAMP=%date:~-4,4%-%date:~-7,2%-%date:~-10,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "LOGFILE=results\logs\batch_run_e5_reuters_%TIMESTAMP%.log"
set "ADAPTER_ROOT=results\adapters"

set "WANDB_API_KEY=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"
set "WANDB_PROJECT=BSC Thesis"
set "WANDB_WATCH=false"



echo.
echo [1/6] Running BASELINE (Mean + Truncated)...
powershell -Command "python main.py --model e5_small --dataset darkreddit --epochs 0 --suffix _base --pooling mean | Tee-Object -FilePath '%LOGFILE%' -Append"

echo.
echo [2/6] Running BASELINE (GeM + Truncated)...
powershell -Command "python main.py --model e5_small --dataset darkreddit --epochs 0 --suffix _base --pooling gmp | Tee-Object -FilePath '%LOGFILE%' -Append"

echo.
echo [3/6] Running BASELINE (Mean + Chunking)...
powershell -Command "python main.py --model e5_small --dataset darkreddit --epochs 0 --suffix _base --pooling mean --chunking | Tee-Object -FilePath '%LOGFILE%' -Append"


echo.
echo ========================================================
echo   ALL EXPERIMENTS COMPLETED FOR %DATASET%
echo   Full log available at: %LOGFILE%
echo ========================================================
pause