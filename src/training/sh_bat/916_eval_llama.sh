#!/bin/bash
#SBATCH -J 916_Eval_Llama
#SBATCH -A p71186
#SBATCH -t 08:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:2
#SBATCH --output=results/logs/916_Eval_Llama_%j.out
#SBATCH --error=results/logs/916_Eval_Llama_%j.err

export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "=== STARTING LLAMA A/B EVALUATION ==="

# --- LLAMA 2 ---
echo "[1/8] Llama 2 | Reuters | Config A"
python -u main.py --model llama2 --dataset reuters --lora --suffix "_ConfigA"

echo "[2/8] Llama 2 | Reuters | Config B"
python -u main.py --model llama2 --dataset reuters --lora --suffix "_ConfigB"

echo "[3/8] Llama 2 | DarkReddit | Config A"
python -u main.py --model llama2 --dataset darkreddit --lora --suffix "_ConfigA"

echo "[4/8] Llama 2 | DarkReddit | Config B"
python -u main.py --model llama2 --dataset darkreddit --lora --suffix "_ConfigB"

# --- LLAMA 3 ---
echo "[5/8] Llama 3 | Reuters | Config A"
python -u main.py --model llama3 --dataset reuters --lora --suffix "_ConfigA"

echo "[6/8] Llama 3 | Reuters | Config B"
python -u main.py --model llama3 --dataset reuters --lora --suffix "_ConfigB"

echo "[7/8] Llama 3 | DarkReddit | Config A"
python -u main.py --model llama3 --dataset darkreddit --lora --suffix "_ConfigA"

echo "[8/8] Llama 3 | DarkReddit | Config B"
python -u main.py --model llama3 --dataset darkreddit --lora --suffix "_ConfigB"

echo "=== LLAMA EVALUATION COMPLETE ==="