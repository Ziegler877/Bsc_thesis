@echo off
chcp 65001 >nul
setlocal

:: ====================================================
:: SETUP LOGGING & VARIABLES
:: ====================================================
set "WANDB_API_KEY=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"
set "WANDB_PROJECT=BSC Thesis"
set "WANDB_WATCH=false"

:: Create a safe directory for logs if it doesn't exist
if not exist "results\logs" mkdir "results\logs"

:: 1. CHOOSE YOUR LOG MODE:
set "TIMESTAMP=%date:~-4,4%-%date:~-7,2%-%date:~-10,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "LOGFILE=results\logs\forced_dynamic_baseline_%TIMESTAMP%.log"

echo ========================================================
echo === ZERO-SHOT BASELINE: E5 FORCED DYNAMIC POOLING ===
echo Log file target: %LOGFILE%
echo ========================================================

:: ====================================================
:: RUN EXPERIMENTS AND REDIRECT OUTPUT
:: ====================================================

::echo [1/4] EVAL: E5-Small ^| DarkReddit ^| DYNAMIC...
::python main.py --model e5_small --dataset darkreddit --pooling dynamic >> "%LOGFILE%" 2>&1

::echo [2/4] EVAL: E5-Small ^| Reuters ^| DYNAMIC...
::python main.py --model e5_small --dataset reuters --pooling dynamic >> "%LOGFILE%" 2>&1

echo [3/4] EVAL: E5-Large ^| DarkReddit ^| DYNAMIC...
python main.py --model e5_large --dataset darkreddit --pooling dynamic >> "%LOGFILE%" 2>&1

::echo [4/4] EVAL: E5-Large ^| Reuters ^| DYNAMIC...
::python main.py --model e5_large --dataset reuters --pooling dynamic >> "%LOGFILE%" 2>&1

echo ========================================================
echo === ALL JOBS COMPLETE. See %LOGFILE% for details. ===
echo ========================================================
pause