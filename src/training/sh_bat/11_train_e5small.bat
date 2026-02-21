@echo off
echo =================================================
echo === STARTING LOCAL TRAINING (TRIPLET LOSS) ===
echo =================================================

REM --- WANDB SETUP ---
REM Tell Hugging Face exactly where to send the data
set WANDB_API_KEY=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
set WANDB_PROJECT=BSC Thesis

REM 1. Activate the Python virtual environment
call .venv\Scripts\activate.bat

REM 2. Run Reuters Training
echo.
echo [1/2] Training E5-Small on REUTERS (Batch: 16)
python src\training\train.py ^
    --model e5_small ^
    --dataset reuters ^
    --epochs 5 ^
    --patience 3 ^
    --batch_size 16

REM 3. Run DarkReddit Training
echo.
echo [2/2] Training E5-Small on DARKREDDIT (Batch: 4)
python src\training\train.py ^
    --model e5_small ^
    --dataset darkreddit ^
    --epochs 100 ^
    --patience 3 ^
    --batch_size 16


echo.
echo === LOCAL TRAINING COMPLETED ===
pause