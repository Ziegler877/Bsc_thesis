#!/bin/bash
#SBATCH -J 919_eval
#SBATCH -A p71186
#SBATCH -t 24:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:2
#SBATCH --output=results/logs/919_EVAL_%j.out
#SBATCH --error=results/logs/919_EVAL_%j.err

export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "============================================"
echo "=== THESIS EVALUATION: ALL TRAINED MODELS ==="
echo "============================================"

# --- E5 SMALL ---
echo "[1/8] EVAL: E5-Small | Reuters | CHUNKED"
python -u main.py --model e5_small --dataset reuters --lora --chunking

echo "[2/8] EVAL: E5-Small | DarkReddit | CHUNKED"
python -u main.py --model e5_small --dataset darkreddit --lora --chunking

# --- E5 LARGE ---
echo "[3/8] EVAL: E5-Large | Reuters | MEAN"
python -u main.py --model e5_large --dataset reuters --lora --pooling mean

echo "[4/8] EVAL: E5-Large | DarkReddit | MEAN"
python -u main.py --model e5_large --dataset darkreddit --lora --pooling mean

# --- LLAMA 2 ---
echo "[5/8] EVAL: Llama 2 | Reuters | MEAN"
python -u main.py --model llama2 --dataset reuters --lora --pooling mean

echo "[6/8] EVAL: Llama 2 | DarkReddit | MEAN"
python -u main.py --model llama2 --dataset darkreddit --lora --pooling mean

# --- LLAMA 3 ---
echo "[7/8] EVAL: Llama 3 | Reuters | MEAN"
python -u main.py --model llama3 --dataset reuters --lora --pooling mean

echo "[8/8] EVAL: Llama 3 | DarkReddit | DYNAMIC"
python -u main.py --model llama3 --dataset darkreddit --lora --pooling dynamic

echo "============================================"
echo "=== ALL EVALUATIONS COMPLETE ==="
echo "============================================"