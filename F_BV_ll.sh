#!/bin/bash
#SBATCH -J F_BV_ll
#SBATCH -A p71186
#SBATCH -t 23:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:2
#SBATCH --output=results/logs/F_BV_ll_%j.out
#SBATCH --error=results/logs/F_BV_ll_%j.err

export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "============================================"
echo "=== LLAMA BASELINE VARIATIONS EVALUATION ==="
echo "============================================"

# ---------------------------------------
# LLAMA 2 | REUTERS
# ---------------------------------------
echo "[1/16] Llama 2 | Reuters | NORMAL"
python -u main.py --model llama2 --dataset reuters

echo "[2/16] Llama 2 | Reuters | CHUNKED"
python -u main.py --model llama2 --dataset reuters --chunking

echo "[3/16] Llama 2 | Reuters | SUB 5"
python -u main.py --model llama2 --dataset reuters --subset 5

echo "[4/16] Llama 2 | Reuters | GMP"
python -u main.py --model llama2 --dataset reuters --pooling gmp

# ---------------------------------------
# LLAMA 2 | DARKREDDIT
# ---------------------------------------
echo "[5/16] Llama 2 | DarkReddit | NORMAL"
python -u main.py --model llama2 --dataset darkreddit

echo "[6/16] Llama 2 | DarkReddit | CHUNKED"
python -u main.py --model llama2 --dataset darkreddit --chunking

echo "[7/16] Llama 2 | DarkReddit | SUB 5"
python -u main.py --model llama2 --dataset darkreddit --subset 5

echo "[8/16] Llama 2 | DarkReddit | GMP"
python -u main.py --model llama2 --dataset darkreddit --pooling gmp

# ---------------------------------------
# LLAMA 3 | REUTERS
# ---------------------------------------
echo "[9/16] Llama 3 | Reuters | NORMAL"
python -u main.py --model llama3 --dataset reuters

echo "[10/16] Llama 3 | Reuters | CHUNKED"
python -u main.py --model llama3 --dataset reuters --chunking

echo "[11/16] Llama 3 | Reuters | SUB 5"
python -u main.py --model llama3 --dataset reuters --subset 5

echo "[12/16] Llama 3 | Reuters | GMP"
python -u main.py --model llama3 --dataset reuters --pooling gmp

# ---------------------------------------
# LLAMA 3 | DARKREDDIT
# ---------------------------------------
echo "[13/16] Llama 3 | DarkReddit | NORMAL"
python -u main.py --model llama3 --dataset darkreddit

echo "[14/16] Llama 3 | DarkReddit | CHUNKED"
python -u main.py --model llama3 --dataset darkreddit --chunking

echo "[15/16] Llama 3 | DarkReddit | SUB 5"
python -u main.py --model llama3 --dataset darkreddit --subset 5

echo "[16/16] Llama 3 | DarkReddit | GMP"
python -u main.py --model llama3 --dataset darkreddit --pooling gmp

echo "=== LLAMA BASELINES COMPLETE ==="