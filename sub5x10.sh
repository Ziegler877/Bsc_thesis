#!/bin/bash
#SBATCH -J sub5x10
#SBATCH -A p71186
#SBATCH -t 24:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/10x_Sub5_Eval_%j.out
#SBATCH --error=results/logs/10x_Sub5_Eval_%j.err

export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "=========================================================="
echo "=== THESIS EVALUATION: 10x SUB-5 FOR ALL TRAINED MODELS ==="
echo "=========================================================="

# Loop 10 times
for i in {1..10}; do
    echo "============================================"
    echo "          STARTING ITERATION $i OF 10       "
    echo "============================================"

    # --- E5 SMALL ---
    echo "[$i/10] EVAL: E5-Small | Reuters | CHUNKED | SUB 5"
    python -u main.py --model e5_small --dataset reuters --lora --chunking --subset 5 --suffix "_run${i}"

    echo "[$i/10] EVAL: E5-Small | DarkReddit | CHUNKED | SUB 5"
    python -u main.py --model e5_small --dataset darkreddit --lora --chunking --subset 5 --suffix "_run${i}"

    # --- E5 LARGE ---
    echo "[$i/10] EVAL: E5-Large | Reuters | MEAN | SUB 5"
    python -u main.py --model e5_large --dataset reuters --lora --pooling mean --subset 5 --suffix "_run${i}"

    echo "[$i/10] EVAL: E5-Large | DarkReddit | MEAN | SUB 5"
    python -u main.py --model e5_large --dataset darkreddit --lora --pooling mean --subset 5 --suffix "_run${i}"

    # --- LLAMA 2 ---
    echo "[$i/10] EVAL: Llama 2 | Reuters | MEAN | SUB 5"
    python -u main.py --model llama2 --dataset reuters --lora --pooling mean --subset 5 --suffix "_run${i}"

    echo "[$i/10] EVAL: Llama 2 | DarkReddit | MEAN | SUB 5"
    python -u main.py --model llama2 --dataset darkreddit --lora --pooling mean --subset 5 --suffix "_run${i}"

    # --- LLAMA 3 ---
    echo "[$i/10] EVAL: Llama 3 | Reuters | MEAN | SUB 5"
    python -u main.py --model llama3 --dataset reuters --lora --pooling mean --subset 5 --suffix "_run${i}"

    echo "[$i/10] EVAL: Llama 3 | DarkReddit | DYNAMIC | SUB 5"
    python -u main.py --model llama3 --dataset darkreddit --lora --pooling dynamic --subset 5 --suffix "_run${i}"

done

echo "============================================"
echo "=== ALL 10 ITERATIONS COMPLETE ==="
echo "============================================"