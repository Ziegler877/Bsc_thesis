#!/bin/bash
#SBATCH -J 08_Subset_5
#SBATCH -A p71186
#SBATCH -t 01:00:00
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/08_Subset_%j.out
#SBATCH --error=results/logs/08_Subset_%j.err

# --- SETUP ---
export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "=== STARTING SUBSET TEST (5 Texts/Author) ==="
echo "   Dataset: Reuters (Train Subsampled)"

# 1. E5 Small
echo "[1/4] E5 Small | Subset: 5"
python -u main.py --model e5_small --dataset reuters --epochs 0 --pooling mean --subset 5

# 2. E5 Large
echo "[2/4] E5 Large | Subset: 5"
python -u main.py --model e5_large --dataset reuters --epochs 0 --pooling mean --subset 5

# 3. Llama 2
echo "[3/4] Llama 2 | Subset: 5"
python -u main.py --model llama2 --dataset reuters --epochs 0 --pooling mean --subset 5

# 4. Llama 3
#echo "[4/4] Llama 3 | Subset: 5"
#python -u main.py --model llama3 --dataset reuters --epochs 0 --pooling mean --subset 5

#echo "=== SUBSET TEST COMPLETE ==="