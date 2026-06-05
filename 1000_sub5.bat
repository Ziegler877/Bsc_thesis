@echo off
echo ==========================================
echo    STARTING SUB-5 EVALUATIONS (LOCAL)
echo ==========================================

REM Set Weights & Biases Variables
set WANDB_API_KEY=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
set WANDB_PROJECT="BSC Thesis"

REM Activate the virtual environment
REM Adjust this path if your .venv is in a different directory!
call .venv\Scripts\activate

echo.
echo === Running E5 Small ===
python main.py --model e5_small --dataset reuters --subset 5 --pooling mean
python main.py --model e5_small --dataset darkreddit --subset 5 --pooling mean

echo.
echo === Running E5 Large ===
python main.py --model e5_large --dataset reuters --subset 5 --pooling mean
python main.py --model e5_large --dataset darkreddit --subset 5 --pooling mean

echo.
echo === Running Llama 2 ===
python main.py --model llama2 --dataset reuters --subset 5 --pooling mean
python main.py --model llama2 --dataset darkreddit --subset 5 --pooling mean

echo.
echo === Running Llama 3 ===
python main.py --model llama3 --dataset reuters --subset 5 --pooling mean
python main.py --model llama3 --dataset darkreddit --subset 5 --pooling mean

echo.
echo ==========================================
echo    ALL SUB-5 EVALUATIONS COMPLETE!
echo ==========================================
pause