#!/bin/bash
#SBATCH -J 901_Train_E5S
#SBATCH -A p71186
#SBATCH -t 12:00:00
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/901_Train_E5S_%j.out
#SBATCH --error=results/logs/901_Train_E5S_%j.err

# --- SETUP ---
export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"

module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411

PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "================================================="
echo "=== STARTING E5-SMALL TRIPLET LOSS TRAINING ==="
echo "================================================="

# --- 1. REUTERS ---
echo ""
echo "[1/2] Training E5-Small on REUTERS (Batch: 32)"
python -u src/training/train.py \
    --model e5_small \
    --dataset reuters \
    --epochs 100 \
    --patience 4 \
    --batch_size 32

# --- 2. DARKREDDIT ---
echo ""
echo "[2/2] Training E5-Small on DARKREDDIT (Batch: 32)"
python -u src/training/train.py \
    --model e5_small \
    --dataset darkreddit \
    --epochs 100 \
    --patience 4 \
    --batch_size 32

echo ""
echo "=== 901: E5-SMALL TRAINING COMPLETED ==="