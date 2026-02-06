@echo off
chcp 65001
setlocal

:: ====================================================
:: SETUP LOGGING
:: ====================================================

:: Generate a unique log filename with Timestamp
::set "TIMESTAMP=%date:~-4,4%-%date:~-7,2%-%date:~-10,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%"
::set "TIMESTAMP=%TIMESTAMP: =0%"
::set "LOGFILE=results\logs\batch_run_%TIMESTAMP%.log"

set "WANDB_API_KEY=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"
set "WANDB_PROJECT=BSC Thesis"
set "WANDB_WATCH=false"

echo ========================================================
echo   STARTING BATCH EXPERIMENT (16 CONFIGURATIONS)
echo   Logs will be saved to: %LOGFILE%
echo ========================================================

:: We use PowerShell to 'Tee' the output (Show on Screen + Save to File)
:: The syntax is: powershell -Command "YOUR_COMMAND | Tee-Object -FilePath '%LOGFILE%' -Append"

:: ====================================================
:: 1. E5 SMALL (Baselines)
:: ====================================================

::set "ADAPTER_ROOT=results\adapters"

::echo.
::echo [1/4] Running E5_SMALL on REUTERS (BASE - 0 Epochs)...
::powershell -Command "python main.py --model e5_small --dataset reuters --epochs 0 --suffix _base | Tee-Object -FilePath '%LOGFILE%' -Append"
::echo [1/4] FINISHED.



::echo.
::echo [2/4] Running E5_SMALL on REUTERS (LoRA - 3 Epochs)...
::if exist "%ADAPTER_ROOT%\e5_small_reuters_3ep" (
::    move /Y "%ADAPTER_ROOT%\e5_small_reuters_3ep" "%ADAPTER_ROOT%\e5_small_reuters" > nul
::)

::powershell -Command "python main.py --model e5_small --dataset reuters --lora --epochs 3 --suffix _3ep | Tee-Object -FilePath '%LOGFILE%' -Append"
::if exist "%ADAPTER_ROOT%\e5_small_reuters" (
::    move /Y "%ADAPTER_ROOT%\e5_small_reuters" "%ADAPTER_ROOT%\e5_small_reuters_3ep" > nul
::)
::echo [2/4] FINISHED.


::echo.
::echo [3/4] Running E5_SMALL on DARKREDDIT (BASE - 0 Epochs)...
::powershell -Command "python main.py --model e5_small --dataset darkreddit --epochs 0 --suffix _base | Tee-Object -FilePath '%LOGFILE%' -Append"
::echo [3/4] FINISHED.

::echo.
::echo [4/4] Running E5_SMALL on DARKREDDIT (LoRA - 3 Epochs)...
::if exist "%ADAPTER_ROOT%\e5_small_darkreddit_3ep" (
::    move /Y "%ADAPTER_ROOT%\e5_small_darkreddit_3ep" "%ADAPTER_ROOT%\e5_small_darkreddit" > nul
::)
::powershell -Command "python main.py --model e5_small --dataset darkreddit --lora --epochs 3 --suffix _3ep | Tee-Object -FilePath '%LOGFILE%' -Append"
::if exist "%ADAPTER_ROOT%\e5_small_darkreddit" (
::    move /Y "%ADAPTER_ROOT%\e5_small_darkreddit" "%ADAPTER_ROOT%\e5_small_darkreddit_3ep" > nul
::)
::echo [4/4] FINISHED.


:: ====================================================
:: 2. E5 LARGE (Baselines)
:: ====================================================

::echo.
::echo [5/16] Running E5_LARGE on REUTERS...
::powershell -Command "python main.py --model e5_large --dataset reuters | Tee-Object -FilePath '%LOGFILE%' -Append"
::echo [5/16] FINISHED.

::echo.
::echo [6/16] Running E5_LARGE on REUTERS (LoRA)...
::powershell -Command "python main.py --model e5_large --dataset reuters --lora | Tee-Object -FilePath '%LOGFILE%' -Append"
::echo [6/16] FINISHED.

::echo.
::echo [7/16] Running E5_LARGE on DARKREDDIT...
::powershell -Command "python main.py --model e5_large --dataset darkreddit | Tee-Object -FilePath '%LOGFILE%' -Append"
::echo [7/16] FINISHED.

::echo.
::echo [8/16] Running E5_LARGE on DARKREDDIT (LoRA)...
::powershell -Command "python main.py --model e5_large --dataset darkreddit --lora | Tee-Object -FilePath '%LOGFILE%' -Append"
::echo [8/16] FINISHED.

:: ====================================================
:: FINISH
:: ====================================================

:: CRITICAL: This variable must be set!
set "ADAPTER_ROOT=results\adapters"
set "DATASET=darkreddit"
set "EPOCHS=3"

:: Generate Log File Name
set "TIMESTAMP=%date:~-4,4%-%date:~-7,2%-%date:~-10,2%_%time:~0,2%-%time:~3,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "LOGFILE=results\logs\eval_battle_%TIMESTAMP%.log"

echo ========================================================
echo   STARTING EVALUATION BATTLE (Base vs A, B, C, D)
echo   Dataset: %DATASET%
echo   Logs: %LOGFILE%
echo ========================================================

:: ========================================================
:: 2. BASE MODEL (Zero-Shot)
:: ========================================================
echo.
echo [1/5] Evaluating BASE MODEL (No LoRA)...
powershell -Command "python main.py --model e5_small --dataset %DATASET% --epochs 0 --suffix _BASE | Tee-Object -FilePath '%LOGFILE%' -Append"
echo [1/5] DONE.

:: ========================================================
:: 3. EXPERIMENT A (DoRA + Cosine)
:: ========================================================
echo.
echo [2/5] Evaluating EXP A...

:: Check if folder exists using the variable
if exist "%ADAPTER_ROOT%\e5_small_%DATASET%_expA" (
    echo    Found folder: e5_small_%DATASET%_expA
    :: Move folder so main.py can find it
    move /Y "%ADAPTER_ROOT%\e5_small_%DATASET%_expA" "%ADAPTER_ROOT%\e5_small_%DATASET%" > nul
) else (
    echo [ERROR] Folder e5_small_%DATASET%_expA not found! Skipping.
    goto :SKIP_A
)

:: Run Evaluation
powershell -Command "python main.py --model e5_small --dataset %DATASET% --lora --epochs %EPOCHS% --suffix _expA | Tee-Object -FilePath '%LOGFILE%' -Append"

:: Move folder back to safe storage
move /Y "%ADAPTER_ROOT%\e5_small_%DATASET%" "%ADAPTER_ROOT%\e5_small_%DATASET%_expA" > nul

:SKIP_A
echo [2/5] DONE.

:: ========================================================
:: 4. EXPERIMENT B (Dropout + Rank 16)
:: ========================================================
echo.
echo [3/5] Evaluating EXP B...

if exist "%ADAPTER_ROOT%\e5_small_%DATASET%_expB" (
    echo    Found folder: e5_small_%DATASET%_expB
    move /Y "%ADAPTER_ROOT%\e5_small_%DATASET%_expB" "%ADAPTER_ROOT%\e5_small_%DATASET%" > nul
) else (
    echo [ERROR] Folder e5_small_%DATASET%_expB not found! Skipping.
    goto :SKIP_B
)

powershell -Command "python main.py --model e5_small --dataset %DATASET% --lora --epochs %EPOCHS% --suffix _expB | Tee-Object -FilePath '%LOGFILE%' -Append"

move /Y "%ADAPTER_ROOT%\e5_small_%DATASET%" "%ADAPTER_ROOT%\e5_small_%DATASET%_expB" > nul

:SKIP_B
echo [3/5] DONE.

:: ========================================================
:: 5. EXPERIMENT C (Bias Tuning)
:: ========================================================
echo.
echo [4/5] Evaluating EXP C...

if exist "%ADAPTER_ROOT%\e5_small_%DATASET%_expC" (
    echo    Found folder: e5_small_%DATASET%_expC
    move /Y "%ADAPTER_ROOT%\e5_small_%DATASET%_expC" "%ADAPTER_ROOT%\e5_small_%DATASET%" > nul
) else (
    echo [ERROR] Folder e5_small_%DATASET%_expC not found! Skipping.
    goto :SKIP_C
)

powershell -Command "python main.py --model e5_small --dataset %DATASET% --lora --epochs %EPOCHS% --suffix _expC | Tee-Object -FilePath '%LOGFILE%' -Append"

move /Y "%ADAPTER_ROOT%\e5_small_%DATASET%" "%ADAPTER_ROOT%\e5_small_%DATASET%_expC" > nul

:SKIP_C
echo [4/5] DONE.

:: ========================================================
:: 6. EXPERIMENT D (Combo)
:: ========================================================
echo.
echo [5/5] Evaluating EXP D...

if exist "%ADAPTER_ROOT%\e5_small_%DATASET%_expD" (
    echo    Found folder: e5_small_%DATASET%_expD
    move /Y "%ADAPTER_ROOT%\e5_small_%DATASET%_expD" "%ADAPTER_ROOT%\e5_small_%DATASET%" > nul
) else (
    echo [ERROR] Folder e5_small_%DATASET%_expD not found! Skipping.
    goto :SKIP_D
)

powershell -Command "python main.py --model e5_small --dataset %DATASET% --lora --epochs %EPOCHS% --suffix _expD | Tee-Object -FilePath '%LOGFILE%' -Append"

move /Y "%ADAPTER_ROOT%\e5_small_%DATASET%" "%ADAPTER_ROOT%\e5_small_%DATASET%_expD" > nul

:SKIP_D
echo [5/5] DONE.


echo.
echo ========================================================
echo   ALL EXPERIMENTS COMPLETED FOR %DATASET%
echo   Full log available at: %LOGFILE%
echo ========================================================
pause