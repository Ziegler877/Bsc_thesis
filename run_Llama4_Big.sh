#!/bin/bash
#SBATCH -J Llama4_Big_Run
#SBATCH -A p71186
#SBATCH -t 02:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:2
#SBATCH --output=results/logs/Llama4_Zen3_%j.out
#SBATCH --error=results/logs/Llama4_Zen3_%j.err

# 1. Load Modules
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411

# 2. Define Project Path
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "=================================================="
echo "   STARTING LLAMA 4 RUN (Zen3 A100 - High RAM)"
echo "   Date: $(date)"
echo "=================================================="

# ------------------------------------------------
# LLAMA 4 BASE MODEL EVALUATION
# ------------------------------------------------
echo ""
echo ">>> [1/2] Evaluating Llama 4 BASE (Reuters)..."
python -u main.py --model llama4 --dataset reuters --device cuda

echo ""
echo ">>> [2/2] Evaluating Llama 4 BASE (DarkReddit)..."
python -u main.py --model llama4 --dataset darkreddit --device cuda

echo ""
echo "=== LLAMA 4 FINISHED ==="