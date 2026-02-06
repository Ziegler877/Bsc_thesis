#!/bin/bash
#SBATCH -J E5_Marathon
#SBATCH -A p71186
#SBATCH -t 12:00:00
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/E5_Marathon_%j.out
#SBATCH --error=results/logs/E5_Marathon_%j.err

# ========================================================
# 1. SETUP ENVIRONMENT
# ========================================================
export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"

# Load Modules
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411

# Define Project Path & Activate Venv
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

# Define Adapter Path
ADAPTER_ROOT="results/adapters"

echo "========================================================"
echo "   STARTING LINEAR MARATHON (Cluster Mode)"
echo "   Date: $(date)"
echo "========================================================"

# ========================================================
# DARKREDDIT - EPOCH 2
# ========================================================

echo ""
echo "[DarkReddit] Exp A (DoRA) - 2 Epochs..."
# Using batch_size 8 for DoRA to prevent OOM
python -u src/train_lora.py --model e5_small --dataset darkreddit --epochs 2 --batch_size 8 --use_dora --lr_scheduler cosine
mv -f "$ADAPTER_ROOT/e5_small_darkreddit_2ep" "$ADAPTER_ROOT/e5_small_darkreddit_expA_2ep"

echo ""
echo "[DarkReddit] Exp B (Dropout) - 2 Epochs..."
python -u src/train_lora.py --model e5_small --dataset darkreddit --epochs 2 --batch_size 16 --lora_dropout 0.3 --r 16 --lora_alpha 32
mv -f "$ADAPTER_ROOT/e5_small_darkreddit_2ep" "$ADAPTER_ROOT/e5_small_darkreddit_expB_2ep"


# ========================================================
# DARKREDDIT - EPOCH 3
# ========================================================

echo ""
echo "[DarkReddit] Exp A (DoRA) - 3 Epochs..."
python -u src/train_lora.py --model e5_small --dataset darkreddit --epochs 3 --batch_size 8 --use_dora --lr_scheduler cosine
mv -f "$ADAPTER_ROOT/e5_small_darkreddit_3ep" "$ADAPTER_ROOT/e5_small_darkreddit_expA_3ep"

echo ""
echo "[DarkReddit] Exp B (Dropout) - 3 Epochs..."
python -u src/train_lora.py --model e5_small --dataset darkreddit --epochs 3 --batch_size 16 --lora_dropout 0.3 --r 16 --lora_alpha 32
mv -f "$ADAPTER_ROOT/e5_small_darkreddit_3ep" "$ADAPTER_ROOT/e5_small_darkreddit_expB_3ep"


# ========================================================
# DARKREDDIT - EPOCH 4
# ========================================================

echo ""
echo "[DarkReddit] Exp A (DoRA) - 4 Epochs..."
python -u src/train_lora.py --model e5_small --dataset darkreddit --epochs 4 --batch_size 8 --use_dora --lr_scheduler cosine
mv -f "$ADAPTER_ROOT/e5_small_darkreddit_4ep" "$ADAPTER_ROOT/e5_small_darkreddit_expA_4ep"

echo ""
echo "[DarkReddit] Exp B (Dropout) - 4 Epochs..."
python -u src/train_lora.py --model e5_small --dataset darkreddit --epochs 4 --batch_size 16 --lora_dropout 0.3 --r 16 --lora_alpha 32
mv -f "$ADAPTER_ROOT/e5_small_darkreddit_4ep" "$ADAPTER_ROOT/e5_small_darkreddit_expB_4ep"


# ========================================================
# DARKREDDIT - EPOCH 5
# ========================================================

echo ""
echo "[DarkReddit] Exp A (DoRA) - 5 Epochs..."
python -u src/train_lora.py --model e5_small --dataset darkreddit --epochs 5 --batch_size 8 --use_dora --lr_scheduler cosine
mv -f "$ADAPTER_ROOT/e5_small_darkreddit_5ep" "$ADAPTER_ROOT/e5_small_darkreddit_expA_5ep"

echo ""
echo "[DarkReddit] Exp B (Dropout) - 5 Epochs..."
python -u src/train_lora.py --model e5_small --dataset darkreddit --epochs 5 --batch_size 16 --lora_dropout 0.3 --r 16 --lora_alpha 32
mv -f "$ADAPTER_ROOT/e5_small_darkreddit_5ep" "$ADAPTER_ROOT/e5_small_darkreddit_expB_5ep"


echo ""
echo "========================================================"
echo "   MARATHON DONE."
echo "========================================================"