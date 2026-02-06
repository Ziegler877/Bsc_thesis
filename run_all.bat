@echo off
chcp 65001
setlocal

:: ====================================================
:: SETUP LOGGING
:: ====================================================

:: Generate a unique log filename with Timestamp
set "TIMESTAMP=%date:~-4,4%-%date:~-7,2%-%date:~-10,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "LOGFILE=results\logs\batch_run_%TIMESTAMP%.log"

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

set "ADAPTER_ROOT=results\adapters"

::echo.
::echo [1/4] Running E5_SMALL on REUTERS (BASE - 0 Epochs)...
::powershell -Command "python main.py --model e5_small --dataset reuters --epochs 0 --suffix _base | Tee-Object -FilePath '%LOGFILE%' -Append"
::echo [1/4] FINISHED.








echo.
echo [2/4] Running E5_SMALL on REUTERS (LoRA - 2 Epochs)...
if exist "%ADAPTER_ROOT%\e5_small_reuters_2ep" (
    move /Y "%ADAPTER_ROOT%\e5_small_reuters_2ep" "%ADAPTER_ROOT%\e5_small_reuters" > nul
)

powershell -Command "python main.py --model e5_small --dataset reuters --lora --epochs 2 --suffix _2ep | Tee-Object -FilePath '%LOGFILE%' -Append"
if exist "%ADAPTER_ROOT%\e5_small_reuters" (
    move /Y "%ADAPTER_ROOT%\e5_small_reuters" "%ADAPTER_ROOT%\e5_small_reuters_2ep" > nul
)
echo [2/4] FINISHED.




echo.
echo [4/4] Running E5_SMALL on DARKREDDIT (LoRA - 2 Epochs)...
if exist "%ADAPTER_ROOT%\e5_small_darkreddit_2ep" (
    move /Y "%ADAPTER_ROOT%\e5_small_darkreddit_2ep" "%ADAPTER_ROOT%\e5_small_darkreddit" > nul
)
powershell -Command "python main.py --model e5_small --dataset darkreddit --lora --epochs 2 --suffix _2ep | Tee-Object -FilePath '%LOGFILE%' -Append"
if exist "%ADAPTER_ROOT%\e5_small_darkreddit" (
    move /Y "%ADAPTER_ROOT%\e5_small_darkreddit" "%ADAPTER_ROOT%\e5_small_darkreddit_2ep" > nul
)
echo [4/4] FINISHED.










echo.
echo [2/4] Running E5_SMALL on REUTERS (LoRA - 3 Epochs)...
if exist "%ADAPTER_ROOT%\e5_small_reuters_3ep" (
    move /Y "%ADAPTER_ROOT%\e5_small_reuters_3ep" "%ADAPTER_ROOT%\e5_small_reuters" > nul
)

powershell -Command "python main.py --model e5_small --dataset reuters --lora --epochs 3 --suffix _3ep | Tee-Object -FilePath '%LOGFILE%' -Append"
if exist "%ADAPTER_ROOT%\e5_small_reuters" (
    move /Y "%ADAPTER_ROOT%\e5_small_reuters" "%ADAPTER_ROOT%\e5_small_reuters_3ep" > nul
)
echo [2/4] FINISHED.


::echo.
::echo [3/4] Running E5_SMALL on DARKREDDIT (BASE - 0 Epochs)...
::powershell -Command "python main.py --model e5_small --dataset darkreddit --epochs 0 --suffix _base | Tee-Object -FilePath '%LOGFILE%' -Append"
::echo [3/4] FINISHED.

echo.
echo [4/4] Running E5_SMALL on DARKREDDIT (LoRA - 3 Epochs)...
if exist "%ADAPTER_ROOT%\e5_small_darkreddit_3ep" (
    move /Y "%ADAPTER_ROOT%\e5_small_darkreddit_3ep" "%ADAPTER_ROOT%\e5_small_darkreddit" > nul
)
powershell -Command "python main.py --model e5_small --dataset darkreddit --lora --epochs 3 --suffix _3ep | Tee-Object -FilePath '%LOGFILE%' -Append"
if exist "%ADAPTER_ROOT%\e5_small_darkreddit" (
    move /Y "%ADAPTER_ROOT%\e5_small_darkreddit" "%ADAPTER_ROOT%\e5_small_darkreddit_3ep" > nul
)
echo [4/4] FINISHED.


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


echo.
echo [2/4] Running E5_SMALL on REUTERS (LoRA - 4 Epochs)...
if exist "%ADAPTER_ROOT%\e5_small_reuters_4ep" (
    move /Y "%ADAPTER_ROOT%\e5_small_reuters_4ep" "%ADAPTER_ROOT%\e5_small_reuters" > nul
)

powershell -Command "python main.py --model e5_small --dataset reuters --lora --epochs 4 --suffix _4ep | Tee-Object -FilePath '%LOGFILE%' -Append"
if exist "%ADAPTER_ROOT%\e5_small_reuters" (
    move /Y "%ADAPTER_ROOT%\e5_small_reuters" "%ADAPTER_ROOT%\e5_small_reuters_4ep" > nul
)
echo [2/4] FINISHED.




echo.
echo [4/4] Running E5_SMALL on DARKREDDIT (LoRA - 4 Epochs)...
if exist "%ADAPTER_ROOT%\e5_small_darkreddit_4ep" (
    move /Y "%ADAPTER_ROOT%\e5_small_darkreddit_4ep" "%ADAPTER_ROOT%\e5_small_darkreddit" > nul
)
powershell -Command "python main.py --model e5_small --dataset darkreddit --lora --epochs 4 --suffix _4ep | Tee-Object -FilePath '%LOGFILE%' -Append"
if exist "%ADAPTER_ROOT%\e5_small_darkreddit" (
    move /Y "%ADAPTER_ROOT%\e5_small_darkreddit" "%ADAPTER_ROOT%\e5_small_darkreddit_4ep" > nul
)
echo [4/4] FINISHED.









echo.
echo [2/4] Running E5_SMALL on REUTERS (LoRA - 5 Epochs)...
if exist "%ADAPTER_ROOT%\e5_small_reuters_5ep" (
    move /Y "%ADAPTER_ROOT%\e5_small_reuters_5ep" "%ADAPTER_ROOT%\e5_small_reuters" > nul
)

powershell -Command "python main.py --model e5_small --dataset reuters --lora --epochs 5 --suffix _5ep | Tee-Object -FilePath '%LOGFILE%' -Append"
if exist "%ADAPTER_ROOT%\e5_small_reuters" (
    move /Y "%ADAPTER_ROOT%\e5_small_reuters" "%ADAPTER_ROOT%\e5_small_reuters_5ep" > nul
)
echo [2/4] FINISHED.




echo.
echo [4/4] Running E5_SMALL on DARKREDDIT (LoRA - 5 Epochs)...
if exist "%ADAPTER_ROOT%\e5_small_darkreddit_5ep" (
    move /Y "%ADAPTER_ROOT%\e5_small_darkreddit_5ep" "%ADAPTER_ROOT%\e5_small_darkreddit" > nul
)
powershell -Command "python main.py --model e5_small --dataset darkreddit --lora --epochs 5 --suffix _5ep | Tee-Object -FilePath '%LOGFILE%' -Append"
if exist "%ADAPTER_ROOT%\e5_small_darkreddit" (
    move /Y "%ADAPTER_ROOT%\e5_small_darkreddit" "%ADAPTER_ROOT%\e5_small_darkreddit_5ep" > nul
)
echo [4/4] FINISHED.





echo.
echo [2/4] Running E5_SMALL on REUTERS (LoRA - 6 Epochs)...
if exist "%ADAPTER_ROOT%\e5_small_reuters_6ep" (
    move /Y "%ADAPTER_ROOT%\e5_small_reuters_6ep" "%ADAPTER_ROOT%\e5_small_reuters" > nul
)

powershell -Command "python main.py --model e5_small --dataset reuters --lora --epochs 6 --suffix _6ep | Tee-Object -FilePath '%LOGFILE%' -Append"
if exist "%ADAPTER_ROOT%\e5_small_reuters" (
    move /Y "%ADAPTER_ROOT%\e5_small_reuters" "%ADAPTER_ROOT%\e5_small_reuters_6ep" > nul
)
echo [2/4] FINISHED.




echo.
echo [4/4] Running E5_SMALL on DARKREDDIT (LoRA - 6 Epochs)...
if exist "%ADAPTER_ROOT%\e5_small_darkreddit_6ep" (
    move /Y "%ADAPTER_ROOT%\e5_small_darkreddit_6ep" "%ADAPTER_ROOT%\e5_small_darkreddit" > nul
)
powershell -Command "python main.py --model e5_small --dataset darkreddit --lora --epochs 6 --suffix _6ep | Tee-Object -FilePath '%LOGFILE%' -Append"
if exist "%ADAPTER_ROOT%\e5_small_darkreddit" (
    move /Y "%ADAPTER_ROOT%\e5_small_darkreddit" "%ADAPTER_ROOT%\e5_small_darkreddit_6ep" > nul
)
echo [4/4] FINISHED.

echo.
echo ========================================================
echo   ALL EXPERIMENTS COMPLETED.
echo   Full log available at: %LOGFILE%
echo ========================================================
pause