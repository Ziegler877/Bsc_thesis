#!/bin/bash
#SBATCH -J 00_Base_All
#SBATCH -A p71186
#SBATCH -t 00:20:00
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/00_Base_All_%j.out
#SBATCH --error=results/logs/00_Base_All_%j.err

# --- SETUP ---
export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "=== BASE ALL START (Checking all 4 models) ==="

# 1. Check E5 Small
echo "[Test 1/4] E5 Small..."
python -u main.py --model e5_small --dataset reuters --epochs 0

# 2. Check E5 Large
echo "[Test 2/4] E5 Large..."
python -u main.py --model e5_large --dataset reuters --epochs 0

# 3. Check Llama 2
echo "[Test 3/4] Llama 2..."
python -u main.py --model llama2 --dataset reuters --epochs 0

# 4. Check Llama 3
echo "[Test 4/4] Llama 3..."
python -u main.py --model llama3 --dataset reuters --epochs 0

echo "=== BASE ALL COMPLETE. CHECK WANDB FOR 4 RUNS. ==="