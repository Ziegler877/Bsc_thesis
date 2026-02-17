#!/bin/bash
#SBATCH -J E5_Full_Pipeline
#SBATCH -A p71186
#SBATCH -t 04:00:00
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/E5_Pipeline_%j.out
#SBATCH --error=results/logs/E5_Pipeline_%j.err

# 1. Load Modules
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411

# 2. Project Path
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

# Define Adapter Path
ADAPTER_ROOT="results/adapters"

echo "=================================================="
echo "   STARTING E5 CLUSTER PIPELINE"
echo "   Date: $(date)"
echo "=================================================="

# ==============================================================================
# SECTION 1: E5 SMALL
# ==============================================================================
MODEL="e5_small"
echo ""
echo "--------------------------------------------------"
echo ">>> [1/2] RUNNING PIPELINE FOR: $MODEL"
echo "--------------------------------------------------"

# --- PHASE 1: BASE EVAL ---
echo "   [1.1] Base Evaluation..."
python -u main.py --model $MODEL --dataset reuters --device cuda --suffix _base
python -u main.py --model $MODEL --dataset darkreddit --device cuda --suffix _base

# --- PHASE 2: 3 EPOCHS ---
echo "   [1.2] Training 3 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset reuters --epochs 3 --batch_size 32
python -u src/train_lora.py --model $MODEL --dataset darkreddit --epochs 3 --batch_size 32

# Rename & Copy for Eval
echo "   [System] Renaming adapters to *_3ep..."
mv ${ADAPTER_ROOT}/${MODEL}_reuters ${ADAPTER_ROOT}/${MODEL}_reuters_3ep
mv ${ADAPTER_ROOT}/${MODEL}_darkreddit ${ADAPTER_ROOT}/${MODEL}_darkreddit_3ep

cp -r ${ADAPTER_ROOT}/${MODEL}_reuters_3ep ${ADAPTER_ROOT}/${MODEL}_reuters
cp -r ${ADAPTER_ROOT}/${MODEL}_darkreddit_3ep ${ADAPTER_ROOT}/${MODEL}_darkreddit

echo "   [1.3] Evaluating 3 Epochs..."
python -u main.py --model $MODEL --dataset reuters --device cuda --use_adapter --suffix _3ep
python -u main.py --model $MODEL --dataset darkreddit --device cuda --use_adapter --suffix _3ep

# Cleanup
rm -rf ${ADAPTER_ROOT}/${MODEL}_reuters
rm -rf ${ADAPTER_ROOT}/${MODEL}_darkreddit

# --- PHASE 3: 5 EPOCHS ---
echo "   [1.4] Training 5 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset reuters --epochs 5 --batch_size 32
python -u src/train_lora.py --model $MODEL --dataset darkreddit --epochs 5 --batch_size 32

# Rename & Copy for Eval
echo "   [System] Renaming adapters to *_5ep..."
mv ${ADAPTER_ROOT}/${MODEL}_reuters ${ADAPTER_ROOT}/${MODEL}_reuters_5ep
mv ${ADAPTER_ROOT}/${MODEL}_darkreddit ${ADAPTER_ROOT}/${MODEL}_darkreddit_5ep

cp -r ${ADAPTER_ROOT}/${MODEL}_reuters_5ep ${ADAPTER_ROOT}/${MODEL}_reuters
cp -r ${ADAPTER_ROOT}/${MODEL}_darkreddit_5ep ${ADAPTER_ROOT}/${MODEL}_darkreddit

echo "   [1.5] Evaluating 5 Epochs..."
python -u main.py --model $MODEL --dataset reuters --device cuda --use_adapter --suffix _5ep
python -u main.py --model $MODEL --dataset darkreddit --device cuda --use_adapter --suffix _5ep

# Cleanup
rm -rf ${ADAPTER_ROOT}/${MODEL}_reuters
rm -rf ${ADAPTER_ROOT}/${MODEL}_darkreddit


# ==============================================================================
# SECTION 2: E5 LARGE
# ==============================================================================
MODEL="e5_large"
echo ""
echo "--------------------------------------------------"
echo ">>> [2/2] RUNNING PIPELINE FOR: $MODEL"
echo "--------------------------------------------------"

# --- PHASE 1: BASE EVAL ---
echo "   [2.1] Base Evaluation..."
python -u main.py --model $MODEL --dataset reuters --device cuda --suffix _base
python -u main.py --model $MODEL --dataset darkreddit --device cuda --suffix _base

# --- PHASE 2: 3 EPOCHS ---
echo "   [2.2] Training 3 Epochs..."
# Note: Lower batch size for Large model
python -u src/train_lora.py --model $MODEL --dataset reuters --epochs 3 --batch_size 16
python -u src/train_lora.py --model $MODEL --dataset darkreddit --epochs 3 --batch_size 16

# Rename & Copy
mv ${ADAPTER_ROOT}/${MODEL}_reuters ${ADAPTER_ROOT}/${MODEL}_reuters_3ep
mv ${ADAPTER_ROOT}/${MODEL}_darkreddit ${ADAPTER_ROOT}/${MODEL}_darkreddit_3ep

cp -r ${ADAPTER_ROOT}/${MODEL}_reuters_3ep ${ADAPTER_ROOT}/${MODEL}_reuters
cp -r ${ADAPTER_ROOT}/${MODEL}_darkreddit_3ep ${ADAPTER_ROOT}/${MODEL}_darkreddit

echo "   [2.3] Evaluating 3 Epochs..."
python -u main.py --model $MODEL --dataset reuters --device cuda --use_adapter --suffix _3ep
python -u main.py --model $MODEL --dataset darkreddit --device cuda --use_adapter --suffix _3ep

# Cleanup
rm -rf ${ADAPTER_ROOT}/${MODEL}_reuters
rm -rf ${ADAPTER_ROOT}/${MODEL}_darkreddit

# --- PHASE 3: 5 EPOCHS ---
echo "   [2.4] Training 5 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset reuters --epochs 5 --batch_size 16
python -u src/train_lora.py --model $MODEL --dataset darkreddit --epochs 5 --batch_size 16

# Rename & Copy
mv ${ADAPTER_ROOT}/${MODEL}_reuters ${ADAPTER_ROOT}/${MODEL}_reuters_5ep
mv ${ADAPTER_ROOT}/${MODEL}_darkreddit ${ADAPTER_ROOT}/${MODEL}_darkreddit_5ep

cp -r ${ADAPTER_ROOT}/${MODEL}_reuters_5ep ${ADAPTER_ROOT}/${MODEL}_reuters
cp -r ${ADAPTER_ROOT}/${MODEL}_darkreddit_5ep ${ADAPTER_ROOT}/${MODEL}_darkreddit

echo "   [2.5] Evaluating 5 Epochs..."
python -u main.py --model $MODEL --dataset reuters --device cuda --use_adapter --suffix _5ep
python -u main.py --model $MODEL --dataset darkreddit --device cuda --use_adapter --suffix _5ep

# Cleanup
rm -rf ${ADAPTER_ROOT}/${MODEL}_reuters
rm -rf ${ADAPTER_ROOT}/${MODEL}_darkreddit

echo ""
echo "=== E5 PIPELINE FINISHED ==="
echo "Check results/embeddings/ for all .pt files"