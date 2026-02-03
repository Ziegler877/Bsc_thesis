@echo off
chcp 65001
setlocal enabledelayedexpansion

:: ====================================================
:: SETUP LOGGING & PATHS
:: ====================================================
set "TIMESTAMP=%date:~-4,4%-%date:~-7,2%-%date:~-10,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "LOGFILE=results\logs\E5_Pipeline_%TIMESTAMP%.log"
set "ADAPTER_ROOT=results\adapters"

echo ========================================================
echo   STARTING E5 FULL PIPELINE (Base -> 3ep -> 5ep)
echo   Logs: %LOGFILE%
echo   Cooling down enabled to protect your laptop.
echo ========================================================

:: ====================================================
:: 1. E5 SMALL PIPELINE
:: ====================================================
echo.
echo ----------------------------------------------------
echo [1/2] STARTING E5 SMALL PIPELINE
echo ----------------------------------------------------

:: --- PHASE 1: BASE EVALUATION ---
echo [1.1] E5_SMALL Base Eval...
powershell -Command "python main.py --model e5_small --dataset reuters --suffix _base | Tee-Object -FilePath '%LOGFILE%' -Append"
powershell -Command "python main.py --model e5_small --dataset darkreddit --suffix _base | Tee-Object -FilePath '%LOGFILE%' -Append"

echo [Cooling] 60s cooldown...
timeout /t 60 /nobreak >nul

:: --- PHASE 2: 3 EPOCHS (Train -> Rename -> Eval) ---
echo [1.2] E5_SMALL Training (3 Epochs)...
:: Train Reuters
powershell -Command "python src/train_lora.py --model e5_small --dataset reuters --epochs 3 --batch_size 16 | Tee-Object -FilePath '%LOGFILE%' -Append"
:: Move/Rename
move "%ADAPTER_ROOT%\e5_small_reuters" "%ADAPTER_ROOT%\e5_small_reuters_3ep" >nul
:: Copy back for Eval
xcopy /E /I /Q "%ADAPTER_ROOT%\e5_small_reuters_3ep" "%ADAPTER_ROOT%\e5_small_reuters" >nul
:: Eval
powershell -Command "python main.py --model e5_small --dataset reuters --lora --suffix _3ep | Tee-Object -FilePath '%LOGFILE%' -Append"
:: Cleanup
rmdir /S /Q "%ADAPTER_ROOT%\e5_small_reuters"

echo [Cooling] 60s cooldown...
timeout /t 60 /nobreak >nul

:: Train DarkReddit
powershell -Command "python src/train_lora.py --model e5_small --dataset darkreddit --epochs 3 --batch_size 16 | Tee-Object -FilePath '%LOGFILE%' -Append"
move "%ADAPTER_ROOT%\e5_small_darkreddit" "%ADAPTER_ROOT%\e5_small_darkreddit_3ep" >nul
xcopy /E /I /Q "%ADAPTER_ROOT%\e5_small_darkreddit_3ep" "%ADAPTER_ROOT%\e5_small_darkreddit" >nul
powershell -Command "python main.py --model e5_small --dataset darkreddit --lora --suffix _3ep | Tee-Object -FilePath '%LOGFILE%' -Append"
rmdir /S /Q "%ADAPTER_ROOT%\e5_small_darkreddit"

echo [Cooling] 60s cooldown...
timeout /t 60 /nobreak >nul

:: --- PHASE 3: 5 EPOCHS (Train -> Rename -> Eval) ---
echo [1.3] E5_SMALL Training (5 Epochs)...
:: Train Reuters
powershell -Command "python src/train_lora.py --model e5_small --dataset reuters --epochs 5 --batch_size 16 | Tee-Object -FilePath '%LOGFILE%' -Append"
move "%ADAPTER_ROOT%\e5_small_reuters" "%ADAPTER_ROOT%\e5_small_reuters_5ep" >nul
xcopy /E /I /Q "%ADAPTER_ROOT%\e5_small_reuters_5ep" "%ADAPTER_ROOT%\e5_small_reuters" >nul
powershell -Command "python main.py --model e5_small --dataset reuters --lora --suffix _5ep | Tee-Object -FilePath '%LOGFILE%' -Append"
rmdir /S /Q "%ADAPTER_ROOT%\e5_small_reuters"

echo [Cooling] 60s cooldown...
timeout /t 60 /nobreak >nul

:: Train DarkReddit
powershell -Command "python src/train_lora.py --model e5_small --dataset darkreddit --epochs 5 --batch_size 16 | Tee-Object -FilePath '%LOGFILE%' -Append"
move "%ADAPTER_ROOT%\e5_small_darkreddit" "%ADAPTER_ROOT%\e5_small_darkreddit_5ep" >nul
xcopy /E /I /Q "%ADAPTER_ROOT%\e5_small_darkreddit_5ep" "%ADAPTER_ROOT%\e5_small_darkreddit" >nul
powershell -Command "python main.py --model e5_small --dataset darkreddit --lora --suffix _5ep | Tee-Object -FilePath '%LOGFILE%' -Append"
rmdir /S /Q "%ADAPTER_ROOT%\e5_small_darkreddit"

echo [Cooling] 60s cooldown...
timeout /t 60 /nobreak >nul


:: ====================================================
:: 2. E5 LARGE PIPELINE (Longer Cooldowns)
:: ====================================================
echo.
echo ----------------------------------------------------
echo [2/2] STARTING E5 LARGE PIPELINE
echo ----------------------------------------------------

:: --- PHASE 1: BASE EVALUATION ---
echo [2.1] E5_LARGE Base Eval...
powershell -Command "python main.py --model e5_large --dataset reuters --suffix _base | Tee-Object -FilePath '%LOGFILE%' -Append"
powershell -Command "python main.py --model e5_large --dataset darkreddit --suffix _base | Tee-Object -FilePath '%LOGFILE%' -Append"

echo [Cooling] 3 Minutes cooldown (Large Model)...
timeout /t 180 /nobreak >nul

:: --- PHASE 2: 3 EPOCHS ---
echo [2.2] E5_LARGE Training (3 Epochs)...
:: Train Reuters
powershell -Command "python src/train_lora.py --model e5_large --dataset reuters --epochs 3 --batch_size 8 | Tee-Object -FilePath '%LOGFILE%' -Append"
move "%ADAPTER_ROOT%\e5_large_reuters" "%ADAPTER_ROOT%\e5_large_reuters_3ep" >nul
xcopy /E /I /Q "%ADAPTER_ROOT%\e5_large_reuters_3ep" "%ADAPTER_ROOT%\e5_large_reuters" >nul
powershell -Command "python main.py --model e5_large --dataset reuters --lora --suffix _3ep | Tee-Object -FilePath '%LOGFILE%' -Append"
rmdir /S /Q "%ADAPTER_ROOT%\e5_large_reuters"

echo [Cooling] 3 Minutes cooldown...
timeout /t 180 /nobreak >nul

:: Train DarkReddit
powershell -Command "python src/train_lora.py --model e5_large --dataset darkreddit --epochs 3 --batch_size 8 | Tee-Object -FilePath '%LOGFILE%' -Append"
move "%ADAPTER_ROOT%\e5_large_darkreddit" "%ADAPTER_ROOT%\e5_large_darkreddit_3ep" >nul
xcopy /E /I /Q "%ADAPTER_ROOT%\e5_large_darkreddit_3ep" "%ADAPTER_ROOT%\e5_large_darkreddit" >nul
powershell -Command "python main.py --model e5_large --dataset darkreddit --lora --suffix _3ep | Tee-Object -FilePath '%LOGFILE%' -Append"
rmdir /S /Q "%ADAPTER_ROOT%\e5_large_darkreddit"

echo [Cooling] 3 Minutes cooldown...
timeout /t 180 /nobreak >nul

:: --- PHASE 3: 5 EPOCHS ---
echo [2.3] E5_LARGE Training (5 Epochs)...
:: Train Reuters
powershell -Command "python src/train_lora.py --model e5_large --dataset reuters --epochs 5 --batch_size 8 | Tee-Object -FilePath '%LOGFILE%' -Append"
move "%ADAPTER_ROOT%\e5_large_reuters" "%ADAPTER_ROOT%\e5_large_reuters_5ep" >nul
xcopy /E /I /Q "%ADAPTER_ROOT%\e5_large_reuters_5ep" "%ADAPTER_ROOT%\e5_large_reuters" >nul
powershell -Command "python main.py --model e5_large --dataset reuters --lora --suffix _5ep | Tee-Object -FilePath '%LOGFILE%' -Append"
rmdir /S /Q "%ADAPTER_ROOT%\e5_large_reuters"

echo [Cooling] 3 Minutes cooldown...
timeout /t 180 /nobreak >nul

:: Train DarkReddit
powershell -Command "python src/train_lora.py --model e5_large --dataset darkreddit --epochs 5 --batch_size 8 | Tee-Object -FilePath '%LOGFILE%' -Append"
move "%ADAPTER_ROOT%\e5_large_darkreddit" "%ADAPTER_ROOT%\e5_large_darkreddit_5ep" >nul
xcopy /E /I /Q "%ADAPTER_ROOT%\e5_large_darkreddit_5ep" "%ADAPTER_ROOT%\e5_large_darkreddit" >nul
powershell -Command "python main.py --model e5_large --dataset darkreddit --lora --suffix _5ep | Tee-Object -FilePath '%LOGFILE%' -Append"
rmdir /S /Q "%ADAPTER_ROOT%\e5_large_darkreddit"

:: ====================================================
:: FINISH
:: ====================================================
echo.
echo ========================================================
echo   ALL E5 PIPELINES COMPLETED SUCCESSFULLY.
echo   Log available at: %LOGFILE%
echo ========================================================
pause