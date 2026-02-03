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

echo ========================================================
echo   STARTING BATCH EXPERIMENT (16 CONFIGURATIONS)
echo   Logs will be saved to: %LOGFILE%
echo ========================================================

:: We use PowerShell to 'Tee' the output (Show on Screen + Save to File)
:: The syntax is: powershell -Command "YOUR_COMMAND | Tee-Object -FilePath '%LOGFILE%' -Append"

:: ====================================================
:: 1. E5 SMALL (Baselines)
:: ====================================================

echo.
echo [1/16] Running E5_SMALL on REUTERS...
powershell -Command "python main.py --model e5_small --dataset reuters | Tee-Object -FilePath '%LOGFILE%' -Append"
echo [1/16] FINISHED.

::echo.
::echo [2/16] Running E5_SMALL on REUTERS (LoRA)...
::powershell -Command "python main.py --model e5_small --dataset reuters --lora | Tee-Object -FilePath '%LOGFILE%' -Append"
::echo [2/16] FINISHED.

::echo.
::echo [3/16] Running E5_SMALL on DARKREDDIT...
::powershell -Command "python main.py --model e5_small --dataset darkreddit | Tee-Object -FilePath '%LOGFILE%' -Append"
::echo [3/16] FINISHED.

::echo.
::echo [4/16] Running E5_SMALL on DARKREDDIT (LoRA)...
::powershell -Command "python main.py --model e5_small --dataset darkreddit --lora | Tee-Object -FilePath '%LOGFILE%' -Append"
::echo [4/16] FINISHED.


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
echo ========================================================
echo   ALL EXPERIMENTS COMPLETED.
echo   Full log available at: %LOGFILE%
echo ========================================================
pause