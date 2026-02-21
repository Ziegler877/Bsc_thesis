#!/bin/bash
#SBATCH -J 09_Base_Llama3
#SBATCH -A p71186
#SBATCH -t 12:00:00
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:2
#SBATCH --output=results/logs/09_Base_Llama3_%j.out
#SBATCH --error=results/logs/09_Base_Llama3_%j.err

# --- SETUP ---
export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411

PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "=== STARTING LLAMA 3 BASELINES ==="

# ==========================================
# LLAMA 3 (Reuters)
# ==========================================

echo "[1/5] Llama 3 | Reuters | Mean Pooling (Standard)"
python -u main.py --model llama3 --dataset reuters --epochs 0 --pooling mean

echo "[2/5] Llama 3 | Reuters | GeM Pooling"
python -u main.py --model llama3 --dataset reuters --epochs 0 --pooling gmp

echo "[3/5] Llama 3 | Reuters | Mean Pooling + Chunking"
python -u main.py --model llama3 --dataset reuters --epochs 0 --pooling mean --chunking

# ==========================================
# LLAMA 3 (DarkReddit)
# ==========================================

echo "[4/5] Llama 3 | DarkReddit | Mean Pooling (Standard)"
python -u main.py --model llama3 --dataset darkreddit --epochs 0 --pooling mean

echo "[5/5] Llama 3 | DarkReddit | GeM Pooling"
python -u main.py --model llama3 --dataset darkreddit --epochs 0 --pooling gmp

echo "=== LLAMA 3 BASELINES DONE ==="