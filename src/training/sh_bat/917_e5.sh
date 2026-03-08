#!/bin/bash
#SBATCH -J 917_E5
#SBATCH -A p71186
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:2
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

echo "============================================"
echo "=== E5 TRAINING: OPTIMIZED CONFIGURATIONS ==="
echo "============================================"

echo "[1/4] E5-Small | Reuters | CHUNKED"
python -u src/training/train.py --model e5_small --dataset reuters --epochs 100 --patience 4 --batch_size 16 --chunking

echo "[2/4] E5-Small | DarkReddit | CHUNKED"
python -u src/training/train.py --model e5_small --dataset darkreddit --epochs 100 --patience 4 --batch_size 16 --chunking

echo "[3/4] E5-Large | Reuters | MEAN"
python -u src/training/train.py --model e5_large --dataset reuters --epochs 100 --patience 4 --batch_size 16 --pooling mean

echo "[4/4] E5-Large | DarkReddit | MEAN"
python -u src/training/train.py --model e5_large --dataset darkreddit --epochs 100 --patience 4 --batch_size 16 --pooling mean

echo "=== E5 TRAINING COMPLETE ==="