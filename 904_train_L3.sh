#!/bin/bash
#SBATCH -J 904_Train_L3
#SBATCH -A p71186
#SBATCH -t 12:00:00
#SBATCH --partition=zen3_0512_a100x2  # <-- Zen 3 partition with NVIDIA A100s
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:2                  # <-- Using BOTH A100 GPUs
#SBATCH --output=results/logs/904_Train_L3_%j.out
#SBATCH --error=results/logs/904_Train_L3_%j.err

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
echo "=== STARTING LLAMA-3.1 TRIPLET LOSS TRAINING ==="
echo "================================================="

# --- 1. REUTERS ---
echo ""
echo "[1/2] Training LLAMA-3.1 on REUTERS (Batch: 4, Multi-GPU)"
python -u src/training/train.py \
    --model llama3 \
    --dataset reuters \
    --epochs 100 \
    --patience 4 \
    --batch_size 4

# --- 2. DARKREDDIT ---
echo ""
echo "[2/2] Training LLAMA-3.1 on DARKREDDIT (Batch: 4, Multi-GPU)"
python -u src/training/train.py \
    --model llama3 \
    --dataset darkreddit \
    --epochs 100 \
    --patience 4 \
    --batch_size 4

echo ""
echo "=== 904: LLAMA-3.1 TRAINING COMPLETED ==="