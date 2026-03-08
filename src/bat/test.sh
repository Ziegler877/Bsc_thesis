#!/bin/bash
#SBATCH -J test
#SBATCH -A p71186
#SBATCH -t 39:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:2
#SBATCH --output=results/logs/test_%j.out
#SBATCH --error=results/logs/test_%j.err

export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "================================================="
echo "=== RESUMING FAILED EVALUATIONS + NEW DYNAMIC ==="
echo "================================================="

# ---------------------------------------
# E5 SMALL
# ---------------------------------------
echo "[1/12] E5 Small | Reuters | CHUNKED"
python -u main.py --model e5_small --dataset reuters --chunking || echo "Failed, moving to next..."

echo "[2/12] E5 Small | Reuters | SUB 5"
python -u main.py --model e5_small --dataset reuters --subset 5 || echo "Failed, moving to next..."

echo "[3/12] E5 Small | Reuters | DYNAMIC"
python -u main.py --model e5_small --dataset reuters --pooling dynamic || echo "Failed, moving to next..."

echo "[4/12] E5 Small | DarkReddit | DYNAMIC"
python -u main.py --model e5_small --dataset darkreddit --pooling dynamic || echo "Failed, moving to next..."

# ---------------------------------------
# E5 LARGE
# ---------------------------------------
echo "[5/12] E5 Large | Reuters | CHUNKED"
python -u main.py --model e5_large --dataset reuters --chunking || echo "Failed, moving to next..."

echo "[6/12] E5 Large | DarkReddit | CHUNKED"
python -u main.py --model e5_large --dataset darkreddit --chunking || echo "Failed, moving to next..."

echo "[7/12] E5 Large | DarkReddit | GMP"
python -u main.py --model e5_large --dataset darkreddit --pooling gmp || echo "Failed, moving to next..."

echo "[8/12] E5 Large | Reuters | DYNAMIC"
python -u main.py --model e5_large --dataset reuters --pooling dynamic || echo "Failed, moving to next..."

echo "[9/12] E5 Large | DarkReddit | DYNAMIC"
python -u main.py --model e5_large --dataset darkreddit --pooling dynamic || echo "Failed, moving to next..."

# ---------------------------------------
# LLAMA 3
# ---------------------------------------
echo "[10/12] Llama 3 | DarkReddit | GMP"
python -u main.py --model llama3 --dataset darkreddit --pooling gmp || echo "Failed, moving to next..."

echo "[11/12] Llama 3 | Reuters | DYNAMIC"
python -u main.py --model llama3 --dataset reuters --pooling dynamic || echo "Failed, moving to next..."

echo "[12/12] Llama 3 | DarkReddit | DYNAMIC"
python -u main.py --model llama3 --dataset darkreddit --pooling dynamic || echo "Failed, moving to next..."

echo "=== EVALUATIONS COMPLETE ==="