#!/bin/bash
#SBATCH -J 920_e5
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

# STEP 0: Protect existing adapters
for m in "e5_small" "e5_large"; do
    for d in "reuters" "darkreddit"; do
        if [ -d "results/adapters/${m}_${d}" ]; then
            mv "results/adapters/${m}_${d}" "results/adapters/${m}_${d}_run0"
            echo "Backed up ${m}_${d} to _run0"
        fi
    done
done

# STEP 1: Loop 4 times
for i in {1..4}; do
    echo "=========================================="
    echo "          STARTING E5 RUN $i OF 4         "
    echo "=========================================="

    # --- E5 Small | Reuters ---
    python -u src/training/train.py --model e5_small --dataset reuters --epochs 100 --patience 4 --batch_size 16 --chunking
    python -u main.py --model e5_small --dataset reuters --lora --chunking --suffix "_run${i}"
    mv "results/adapters/e5_small_reuters" "results/adapters/e5_small_reuters_run${i}"

    # --- E5 Small | DarkReddit ---
    python -u src/training/train.py --model e5_small --dataset darkreddit --epochs 100 --patience 4 --batch_size 8 --chunking
    python -u main.py --model e5_small --dataset darkreddit --lora --chunking --suffix "_run${i}"
    mv "results/adapters/e5_small_darkreddit" "results/adapters/e5_small_darkreddit_run${i}"

    # --- E5 Large | Reuters ---
    python -u src/training/train.py --model e5_large --dataset reuters --epochs 100 --patience 4 --batch_size 16 --pooling mean
    python -u main.py --model e5_large --dataset reuters --lora --pooling mean --suffix "_run${i}"
    mv "results/adapters/e5_large_reuters" "results/adapters/e5_large_reuters_run${i}"

    # --- E5 Large | DarkReddit ---
    python -u src/training/train.py --model e5_large --dataset darkreddit --epochs 100 --patience 4 --batch_size 8 --pooling mean
    python -u main.py --model e5_large --dataset darkreddit --lora --pooling mean --suffix "_run${i}"
    mv "results/adapters/e5_large_darkreddit" "results/adapters/e5_large_darkreddit_run${i}"
done

echo "=== E5 TRAINING COMPLETE ==="