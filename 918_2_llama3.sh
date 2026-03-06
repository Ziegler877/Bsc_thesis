#!/bin/bash
#SBATCH -J 918_L3
#SBATCH -A p71186
#SBATCH -t 48:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:2
#SBATCH --output=results/logs/918_L3_%j.out
#SBATCH --error=results/logs/918_L3_%j.err

export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "=== LLAMA 3 TRAINING ==="

echo "[1/2] Llama 3 | Reuters | NORMAL"
python -u src/training/train.py --model llama3 --dataset reuters --epochs 100 --patience 4

echo "[2/2] Llama 3 | DarkReddit | NORMAL"
python -u src/training/train.py --model llama3 --dataset darkreddit --epochs 100 --patience 4

echo "=== LLAMA 3 COMPLETE ==="