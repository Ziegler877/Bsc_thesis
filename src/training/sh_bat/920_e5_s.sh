#!/bin/bash
#SBATCH -J 920_e5_s
#SBATCH -A p71186
#SBATCH -t 48:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:2
#SBATCH --output=results/logs/920_e5_%j.out
#SBATCH --error=results/logs/920_e5_%j.err

export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "============================================"
echo "=== E5 TRAINING LOOP: 4x RUNS PER CONFIG ==="
echo "============================================"



echo "============================================"
echo "=== E5 TRAINING LOOP: 4x RUNS PER CONFIG ==="
echo "============================================"

for i in {1..5}; do
    echo "=========================================="
    echo "          STARTING E5 RUN $i OF 5         "
    echo "=========================================="

    # --- E5 Small | Reuters ---
    python -u src/training/train.py --model e5_small --dataset reuters --epochs 100 --patience 10 --batch_size 16 --chunking
    mv "results/adapters/e5_small_reuters" "results/adapters/e5_small_reuters_run${i}"
    python -u main.py --model e5_small --dataset reuters --lora --chunking --suffix "_run${i}"

    # --- E5 Small | DarkReddit ---
    python -u src/training/train.py --model e5_small --dataset darkreddit --epochs 100 --patience 10 --batch_size 8 --chunking
    mv "results/adapters/e5_small_darkreddit" "results/adapters/e5_small_darkreddit_run${i}"
    python -u main.py --model e5_small --dataset darkreddit --lora --chunking --suffix "_run${i}"

done

echo "=== E5 TRAINING COMPLETE ==="