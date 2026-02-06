#!/bin/bash
#SBATCH -J Llama2_MultiEpoch
#SBATCH -A p71186
#SBATCH -t 06:00:00
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/Llama2_Multi_%j.out
#SBATCH --error=results/logs/Llama2_Multi_%j.err

export WANDB_API_KEY=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
export WANDB_PROJECT=BSC Thesis

# 1. Load Modules
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411

# 2. Define Project Path
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

# Define Adapter Path Variable (makes script cleaner)
ADAPTER_ROOT="results/adapters"

echo "=================================================="
echo "   STARTING LLAMA 2 MULTI-EPOCH PIPELINE"
echo "   Date: $(date)"
echo "=================================================="

# ==============================================================================
# PHASE 1: BASE MODEL (0 Epochs)
# ==============================================================================
echo ""
echo ">>> [1/5] Evaluating BASE Model (No LoRA)..."
# We add --suffix _base to clearly separate these files
python -u main.py --model llama2 --dataset reuters --device cuda --suffix _base
python -u main.py --model llama2 --dataset darkreddit --device cuda --suffix _base

# ==============================================================================
# PHASE 2: TRAIN & EVAL 3 EPOCHS
# ==============================================================================
echo ""
echo ">>> [2/5] Training LoRA (3 Epochs)..."
# 1. Train
python -u src/train_lora.py --model llama2 --dataset reuters --epochs 3 --batch_size 4
python -u src/train_lora.py --model llama2 --dataset darkreddit --epochs 3 --batch_size 4

# 2. Rename the output folder to save it safely (e.g. llama2_reuters -> llama2_reuters_3ep)
echo "   [System] Renaming adapters to *_3ep..."
mv ${ADAPTER_ROOT}/llama2_reuters ${ADAPTER_ROOT}/llama2_reuters_3ep
mv ${ADAPTER_ROOT}/llama2_darkreddit ${ADAPTER_ROOT}/llama2_darkreddit_3ep

# 3. Setup for Evaluation (Copy 3ep back to standard name so main.py finds it)
cp -r ${ADAPTER_ROOT}/llama2_reuters_3ep ${ADAPTER_ROOT}/llama2_reuters
cp -r ${ADAPTER_ROOT}/llama2_darkreddit_3ep ${ADAPTER_ROOT}/llama2_darkreddit

echo ">>> [3/5] Evaluating LoRA (3 Epochs)..."
# Run Eval with --suffix _3ep
python -u main.py --model llama2 --dataset reuters --device cuda --use_adapter --suffix _3ep
python -u main.py --model llama2 --dataset darkreddit --device cuda --use_adapter --suffix _3ep

# 4. Cleanup: Remove the temporary standard folder
rm -rf ${ADAPTER_ROOT}/llama2_reuters
rm -rf ${ADAPTER_ROOT}/llama2_darkreddit

# ==============================================================================
# PHASE 3: TRAIN & EVAL 5 EPOCHS
# ==============================================================================
echo ""
echo ">>> [4/5] Training LoRA (5 Epochs)..."
# 1. Train (Starts fresh from base, goes to 5)
python -u src/train_lora.py --model llama2 --dataset reuters --epochs 5 --batch_size 4
python -u src/train_lora.py --model llama2 --dataset darkreddit --epochs 5 --batch_size 4

# 2. Rename the output folder (llama2_reuters -> llama2_reuters_5ep)
echo "   [System] Renaming adapters to *_5ep..."
mv ${ADAPTER_ROOT}/llama2_reuters ${ADAPTER_ROOT}/llama2_reuters_5ep
mv ${ADAPTER_ROOT}/llama2_darkreddit ${ADAPTER_ROOT}/llama2_darkreddit_5ep

# 3. Setup for Evaluation (Copy 5ep back to standard name)
cp -r ${ADAPTER_ROOT}/llama2_reuters_5ep ${ADAPTER_ROOT}/llama2_reuters
cp -r ${ADAPTER_ROOT}/llama2_darkreddit_5ep ${ADAPTER_ROOT}/llama2_darkreddit

echo ">>> [5/5] Evaluating LoRA (5 Epochs)..."
# Run Eval with --suffix _5ep
python -u main.py --model llama2 --dataset reuters --device cuda --use_adapter --suffix _5ep
python -u main.py --model llama2 --dataset darkreddit --device cuda --use_adapter --suffix _5ep

# 4. Cleanup
rm -rf ${ADAPTER_ROOT}/llama2_reuters
rm -rf ${ADAPTER_ROOT}/llama2_darkreddit

echo ""
echo "=== MULTI-EPOCH PIPELINE FINISHED ==="
echo "Results available in results/embeddings/"
ls -lh results/embeddings/