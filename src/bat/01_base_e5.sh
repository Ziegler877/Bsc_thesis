#!/bin/bash
#SBATCH -J 01_Base_E5
#SBATCH -A p71186
#SBATCH -t 04:00:00
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/01_Base_E5_%j.out
#SBATCH --error=results/logs/01_Base_E5_%j.err

# --- SETUP ---
export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "=== STARTING E5 BASELINES ==="

# ==========================================
# E5 SMALL
# ==========================================

echo "[1/10] E5 Small | Reuters | Mean Pooling (Standard)"
python -u main.py --model e5_small --dataset reuters --epochs 0 --pooling mean

echo "[2/10] E5 Small | Reuters | GeM Pooling"
python -u main.py --model e5_small --dataset reuters --epochs 0 --pooling gmp

echo "[3/10] E5 Small | Reuters | Mean Pooling + Chunking"
python -u main.py --model e5_small --dataset reuters --epochs 0 --pooling mean --chunking

echo "[4/10] E5 Small | DarkReddit | Mean Pooling (Standard)"
python -u main.py --model e5_small --dataset darkreddit --epochs 0 --pooling mean

echo "[5/10] E5 Small | DarkReddit | GeM Pooling"
python -u main.py --model e5_small --dataset darkreddit --epochs 0 --pooling gmp


# ==========================================
# E5 LARGE
# ==========================================

echo "[6/10] E5 Large | Reuters | Mean Pooling (Standard)"
python -u main.py --model e5_large --dataset reuters --epochs 0 --pooling mean

echo "[7/10] E5 Large | Reuters | GeM Pooling"
python -u main.py --model e5_large --dataset reuters --epochs 0 --pooling gmp

echo "[8/10] E5 Large | Reuters | Mean Pooling + Chunking"
python -u main.py --model e5_large --dataset reuters --epochs 0 --pooling mean --chunking

echo "[9/10] E5 Large | DarkReddit | Mean Pooling (Standard)"
python -u main.py --model e5_large --dataset darkreddit --epochs 0 --pooling mean

echo "[10/10] E5 Large | DarkReddit | GeM Pooling"
python -u main.py --model e5_large --dataset darkreddit --epochs 0 --pooling gmp

echo "=== E5 BASELINES DONE ==="