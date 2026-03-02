#!/bin/bash
#SBATCH -J 917_E5
#SBATCH -A p71186
#SBATCH -t 24:00:00
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/917_E5_%j.out
#SBATCH --error=results/logs/917_E5_%j.err

export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "=== E5 TRAINING: CONFIG A & NORMAL ==="

# --- E5 SMALL | REUTERS ---
echo "[1/8] E5-Small | Reuters | Config A (Margin 1.0)"
python -u src/training/train.py --model e5_small --dataset reuters --epochs 100 --patience 4 --batch_size 16 --triplet_margin 1.0 --suffix "_ConfigA"

echo "[2/8] E5-Small | Reuters | NORMAL (Default)"
python -u src/training/train.py --model e5_small --dataset reuters --epochs 100 --patience 4 --batch_size 16 --suffix "_Normal"

# --- E5 SMALL | DARKREDDIT ---
echo "[3/8] E5-Small | DarkReddit | Config A (Margin 1.0)"
python -u src/training/train.py --model e5_small --dataset darkreddit --epochs 100 --patience 4 --batch_size 16 --triplet_margin 1.0 --suffix "_ConfigA"

echo "[4/8] E5-Small | DarkReddit | NORMAL (Default)"
python -u src/training/train.py --model e5_small --dataset darkreddit --epochs 100 --patience 4 --batch_size 16 --suffix "_Normal"

# --- E5 LARGE | REUTERS ---
echo "[5/8] E5-Large | Reuters | Config A (Margin 1.0)"
python -u src/training/train.py --model e5_large --dataset reuters --epochs 100 --patience 4 --batch_size 16 --triplet_margin 1.0 --suffix "_ConfigA"

echo "[6/8] E5-Large | Reuters | NORMAL (Default)"
python -u src/training/train.py --model e5_large --dataset reuters --epochs 100 --patience 4 --batch_size 16 --suffix "_Normal"

# --- E5 LARGE | DARKREDDIT ---
echo "[7/8] E5-Large | DarkReddit | Config A (Margin 1.0)"
python -u src/training/train.py --model e5_large --dataset darkreddit --epochs 100 --patience 4 --batch_size 16 --triplet_margin 1.0 --suffix "_ConfigA"

echo "[8/8] E5-Large | DarkReddit | NORMAL (Default)"
python -u src/training/train.py --model e5_large --dataset darkreddit --epochs 100 --patience 4 --batch_size 16 --suffix "_Normal"

echo "=== E5 COMPLETE ==="