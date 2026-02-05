#!/bin/bash
#SBATCH -J Llama4_Scout_Pipeline
#SBATCH -A p71186
#SBATCH -t 08:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=results/logs/Scout_Pipeline_%j.out
#SBATCH --error=results/logs/Scout_Pipeline_%j.err


export WANDB_API_KEY=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
export WANDB_PROJECT=BSC Thesis

# 1. Load Modules (Ensure these match your Zen3 environment)
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411

# 2. Project Path
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

# Define Adapter Path
ADAPTER_ROOT="results/adapters"
MODEL="llama4_scout"

echo "=================================================="
echo "   STARTING LLAMA 4 SCOUT (17B) PIPELINE"
echo "   Partition: Zen3 A100"
echo "   Date: $(date)"
echo "=================================================="

# ==============================================================================
# PHASE 1: BASE MODEL (0 Epochs)
# ==============================================================================
echo ""
echo ">>> [1/5] Evaluating BASE Model (No LoRA)..."
# We use   to ensure the 17B model fits comfortably in memory
python -u main.py --model $MODEL --dataset reuters --device cuda --suffix _base  
python -u main.py --model $MODEL --dataset darkreddit --device cuda --suffix _base  

# ==============================================================================
# PHASE 2: TRAIN & EVAL 3 EPOCHS
# ==============================================================================
echo ""
echo ">>> [2/5] Training LoRA (3 Epochs)..."
# Batch size 4 is safe for 17B. If A100 has 80GB, you could try 8 or 16.
python -u src/train_lora.py --model $MODEL --dataset reuters --epochs 3 --batch_size 4
python -u src/train_lora.py --model $MODEL --dataset darkreddit --epochs 3 --batch_size 4

# Rename & Copy
echo "   [System] Renaming adapters to *_3ep..."
mv ${ADAPTER_ROOT}/${MODEL}_reuters ${ADAPTER_ROOT}/${MODEL}_reuters_3ep
mv ${ADAPTER_ROOT}/${MODEL}_darkreddit ${ADAPTER_ROOT}/${MODEL}_darkreddit_3ep

cp -r ${ADAPTER_ROOT}/${MODEL}_reuters_3ep ${ADAPTER_ROOT}/${MODEL}_reuters
cp -r ${ADAPTER_ROOT}/${MODEL}_darkreddit_3ep ${ADAPTER_ROOT}/${MODEL}_darkreddit

echo ">>> [3/5] Evaluating LoRA (3 Epochs)..."
python -u main.py --model $MODEL --dataset reuters --device cuda --use_adapter --suffix _3ep  
python -u main.py --model $MODEL --dataset darkreddit --device cuda --use_adapter --suffix _3ep  

# Cleanup
rm -rf ${ADAPTER_ROOT}/${MODEL}_reuters
rm -rf ${ADAPTER_ROOT}/${MODEL}_darkreddit

# ==============================================================================
# PHASE 3: TRAIN & EVAL 5 EPOCHS
# ==============================================================================
echo ""
echo ">>> [4/5] Training LoRA (5 Epochs)..."
python -u src/train_lora.py --model $MODEL --dataset reuters --epochs 5 --batch_size 4
python -u src/train_lora.py --model $MODEL --dataset darkreddit --epochs 5 --batch_size 4

# Rename & Copy
echo "   [System] Renaming adapters to *_5ep..."
mv ${ADAPTER_ROOT}/${MODEL}_reuters ${ADAPTER_ROOT}/${MODEL}_reuters_5ep
mv ${ADAPTER_ROOT}/${MODEL}_darkreddit ${ADAPTER_ROOT}/${MODEL}_darkreddit_5ep

cp -r ${ADAPTER_ROOT}/${MODEL}_reuters_5ep ${ADAPTER_ROOT}/${MODEL}_reuters
cp -r ${ADAPTER_ROOT}/${MODEL}_darkreddit_5ep ${ADAPTER_ROOT}/${MODEL}_darkreddit

echo ">>> [5/5] Evaluating LoRA (5 Epochs)..."
python -u main.py --model $MODEL --dataset reuters --device cuda --use_adapter --suffix _5ep  
python -u main.py --model $MODEL --dataset darkreddit --device cuda --use_adapter --suffix _5ep  

# Cleanup
rm -rf ${ADAPTER_ROOT}/${MODEL}_reuters
rm -rf ${ADAPTER_ROOT}/${MODEL}_darkreddit

echo ""
echo "=== SCOUT PIPELINE FINISHED ==="
echo "Results available in results/embeddings/"