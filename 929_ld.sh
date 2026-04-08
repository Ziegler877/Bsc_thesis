#!/bin/bash
#SBATCH -J 929_ld
#SBATCH -A p71186
#SBATCH -t 72:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:2
#SBATCH --output=results/logs/929_ld_%j.out
#SBATCH --error=results/logs/929_ld_%j.err

export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411

PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "=========================================="
echo "          STARTING E5 RUN 6               "
echo "=========================================="

python -u src/training/train.py --model e5_large --dataset darkreddit --epochs 100 --patience 10 --batch_size 8 --pooling mean
mv "results/adapters/e5_large_darkreddit" "results/adapters/e5_large_darkreddit_run6"
python -u main.py --model e5_large --dataset darkreddit --lora --pooling mean --suffix "_run6"

echo "=== E5 RUN 6 COMPLETE ==="