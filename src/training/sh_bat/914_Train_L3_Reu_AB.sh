#!/bin/bash
#SBATCH -J 914_L3_Reu_AB
#SBATCH -A p71186
#SBATCH -t 44:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:2
#SBATCH --output=results/logs/914_L3_Reu_AB_%j.out
#SBATCH --error=results/logs/914_L3_Reu_AB_%j.err

export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "=== LLAMA 3 REUTERS A/B TESTING ==="

echo "[1/2] Llama 3 | Reuters | Config A (Margin 1.0)"
python -u src/training/train.py --model llama3 --dataset reuters --epochs 100 --patience 4 --batch_size 4 --triplet_margin 1.0

echo "[2/2] Llama 3 | Reuters | Config B (LR 1e-4, Rank 128)"
python -u src/training/train.py --model llama3 --dataset reuters --epochs 100 --patience 4 --batch_size 4 --lr 1e-4 --r 128 --lora_alpha 256

echo "=== COMPLETED ==="