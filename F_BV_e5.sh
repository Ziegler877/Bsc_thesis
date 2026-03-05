#!/bin/bash
#SBATCH -J F_BV_e5
#SBATCH -A p71186
#SBATCH -t 24:00:00
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/F_BV_e5_%j.out
#SBATCH --error=results/logs/F_BV_e5_%j.err

export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "========================================="
echo "=== E5 BASELINE VARIATIONS EVALUATION ==="
echo "========================================="

# ---------------------------------------
# E5 SMALL | REUTERS
# ---------------------------------------
echo "[1/16] E5-Small | Reuters | NORMAL"
python -u main.py --model e5_small --dataset reuters

echo "[2/16] E5-Small | Reuters | CHUNKED"
python -u main.py --model e5_small --dataset reuters --chunking

echo "[3/16] E5-Small | Reuters | SUB 5"
python -u main.py --model e5_small --dataset reuters --subset 5

echo "[4/16] E5-Small | Reuters | GMP"
python -u main.py --model e5_small --dataset reuters --pooling gmp

# ---------------------------------------
# E5 SMALL | DARKREDDIT
# ---------------------------------------
echo "[5/16] E5-Small | DarkReddit | NORMAL"
python -u main.py --model e5_small --dataset darkreddit

echo "[6/16] E5-Small | DarkReddit | CHUNKED"
python -u main.py --model e5_small --dataset darkreddit --chunking

echo "[7/16] E5-Small | DarkReddit | SUB 5"
python -u main.py --model e5_small --dataset darkreddit --subset 5

echo "[8/16] E5-Small | DarkReddit | GMP"
python -u main.py --model e5_small --dataset darkreddit --pooling gmp

# ---------------------------------------
# E5 LARGE | REUTERS
# ---------------------------------------
echo "[9/16] E5-Large | Reuters | NORMAL"
python -u main.py --model e5_large --dataset reuters

echo "[10/16] E5-Large | Reuters | CHUNKED"
python -u main.py --model e5_large --dataset reuters --chunking

echo "[11/16] E5-Large | Reuters | SUB 5"
python -u main.py --model e5_large --dataset reuters --subset 5

echo "[12/16] E5-Large | Reuters | GMP"
python -u main.py --model e5_large --dataset reuters --pooling gmp

# ---------------------------------------
# E5 LARGE | DARKREDDIT
# ---------------------------------------
echo "[13/16] E5-Large | DarkReddit | NORMAL"
python -u main.py --model e5_large --dataset darkreddit

echo "[14/16] E5-Large | DarkReddit | CHUNKED"
python -u main.py --model e5_large --dataset darkreddit --chunking

echo "[15/16] E5-Large | DarkReddit | SUB 5"
python -u main.py --model e5_large --dataset darkreddit --subset 5

echo "[16/16] E5-Large | DarkReddit | GMP"
python -u main.py --model e5_large --dataset darkreddit --pooling gmp

echo "=== E5 BASELINES COMPLETE ==="