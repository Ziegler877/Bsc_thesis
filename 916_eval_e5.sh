#!/bin/bash
#SBATCH -J 916_Eval_E5
#SBATCH -A p71186
#SBATCH -t 04:00:00
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/916_Eval_E5_%j.out
#SBATCH --error=results/logs/916_Eval_E5_%j.err

export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "=== STARTING E5 A/B EVALUATION ==="

# --- E5 SMALL ---
echo "[1/8] E5-Small | Reuters | Config A"
python -u main.py --model e5_small --dataset reuters --lora --suffix "_ConfigA"

echo "[2/8] E5-Small | Reuters | Config B"
python -u main.py --model e5_small --dataset reuters --lora --suffix "_ConfigB"

echo "[3/8] E5-Small | DarkReddit | Config A"
python -u main.py --model e5_small --dataset darkreddit --lora --suffix "_ConfigA"

echo "[4/8] E5-Small | DarkReddit | Config B"
python -u main.py --model e5_small --dataset darkreddit --lora --suffix "_ConfigB"

# --- E5 LARGE ---
echo "[5/8] E5-Large | Reuters | Config A"
python -u main.py --model e5_large --dataset reuters --lora --suffix "_ConfigA"

echo "[6/8] E5-Large | Reuters | Config B"
python -u main.py --model e5_large --dataset reuters --lora --suffix "_ConfigB"

echo "[7/8] E5-Large | DarkReddit | Config A"
python -u main.py --model e5_large --dataset darkreddit --lora --suffix "_ConfigA"

echo "[8/8] E5-Large | DarkReddit | Config B"
python -u main.py --model e5_large --dataset darkreddit --lora --suffix "_ConfigB"

echo "=== E5 EVALUATION COMPLETE ==="