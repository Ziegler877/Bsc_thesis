#!/bin/bash
#SBATCH -J 12_Eval_E5
#SBATCH -A p71186
#SBATCH -t 04:00:00
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/12_Eval_E5_%j.out
#SBATCH --error=results/logs/12_Eval_E5_%j.err

export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "=== STARTING E5 EVALUATION (WITH LORA ADAPTERS) ==="

echo "[1/4] E5-Small | Reuters"
python -u main.py --model e5_small --dataset reuters --lora

echo "[2/4] E5-Small | DarkReddit"
python -u main.py --model e5_small --dataset darkreddit --lora

echo "[3/4] E5-Large | Reuters"
python -u main.py --model e5_large --dataset reuters --lora

echo "[4/4] E5-Large | DarkReddit"
python -u main.py --model e5_large --dataset darkreddit --lora

echo "=== E5 EVALUATION COMPLETE ==="