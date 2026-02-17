#!/bin/bash
#SBATCH -J 07_GeM_All
#SBATCH -A p71186
#SBATCH -t 06:00:00
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/07_GeM_%j.out
#SBATCH --error=results/logs/07_GeM_%j.err

# --- SETUP ---
export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "=== STARTING GeM (Generalized Mean Pooling) RUNS ==="

# ==========================================
# 1. E5 SMALL
# ==========================================
echo "[1/8] E5 Small | Reuters | GeM"
python -u main.py --model e5_small --dataset reuters --epochs 0 --pooling gmp

echo "[2/8] E5 Small | DarkReddit | GeM"
python -u main.py --model e5_small --dataset darkreddit --epochs 0 --pooling gmp


# ==========================================
# 2. E5 LARGE
# ==========================================
echo "[3/8] E5 Large | Reuters | GeM"
python -u main.py --model e5_large --dataset reuters --epochs 0 --pooling gmp

echo "[4/8] E5 Large | DarkReddit | GeM"
python -u main.py --model e5_large --dataset darkreddit --epochs 0 --pooling gmp


# ==========================================
# 3. LLAMA 2
# ==========================================
echo "[5/8] Llama 2 | Reuters | GeM"
python -u main.py --model llama2 --dataset reuters --epochs 0 --pooling gmp

echo "[6/8] Llama 2 | DarkReddit | GeM"
python -u main.py --model llama2 --dataset darkreddit --epochs 0 --pooling gmp


# ==========================================
# 4. LLAMA 3
# ==========================================
echo "[7/8] Llama 3 | Reuters | GeM"
python -u main.py --model llama3 --dataset reuters --epochs 0 --pooling gmp

echo "[8/8] Llama 3 | DarkReddit | GeM"
python -u main.py --model llama3 --dataset darkreddit --epochs 0 --pooling gmp

echo "=== GeM RUNS COMPLETE ==="