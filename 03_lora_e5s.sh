#!/bin/bash
#SBATCH -J 03_LoRA_E5S
#SBATCH -A p71186
#SBATCH -t 12:00:00
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/03_LoRA_E5S_%j.out
#SBATCH --error=results/logs/03_LoRA_E5S_%j.err

# --- SETUP ---
export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

ADAPTER_ROOT="results/adapters"
MODEL="e5_small"

echo "=== STARTING MARATHON: E5 SMALL ==="

# ========================================================
# 1. REUTERS
# ========================================================

echo "--- REUTERS: EPOCH 2 ---"
# Exp A (DoRA)
echo "[Reuters] Exp A (DoRA) - 2 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset reuters --epochs 2 --batch_size 8 --use_dora --lr_scheduler cosine
mv -f "$ADAPTER_ROOT/${MODEL}_reuters_2ep" "$ADAPTER_ROOT/${MODEL}_reuters_expA_2ep"
python -u main.py --model $MODEL --dataset reuters --lora --epochs 2 --suffix "_expA" --pooling mean

# Exp B (Dropout)
echo "[Reuters] Exp B (Dropout) - 2 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset reuters --epochs 2 --batch_size 16 --lora_dropout 0.3 --r 16 --lora_alpha 32
mv -f "$ADAPTER_ROOT/${MODEL}_reuters_2ep" "$ADAPTER_ROOT/${MODEL}_reuters_expB_2ep"
python -u main.py --model $MODEL --dataset reuters --lora --epochs 2 --suffix "_expB" --pooling mean


echo "--- REUTERS: EPOCH 3 ---"
# Exp A (DoRA)
echo "[Reuters] Exp A (DoRA) - 3 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset reuters --epochs 3 --batch_size 8 --use_dora --lr_scheduler cosine
mv -f "$ADAPTER_ROOT/${MODEL}_reuters_3ep" "$ADAPTER_ROOT/${MODEL}_reuters_expA_3ep"
python -u main.py --model $MODEL --dataset reuters --lora --epochs 3 --suffix "_expA" --pooling mean

# Exp B (Dropout)
echo "[Reuters] Exp B (Dropout) - 3 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset reuters --epochs 3 --batch_size 16 --lora_dropout 0.3 --r 16 --lora_alpha 32
mv -f "$ADAPTER_ROOT/${MODEL}_reuters_3ep" "$ADAPTER_ROOT/${MODEL}_reuters_expB_3ep"
python -u main.py --model $MODEL --dataset reuters --lora --epochs 3 --suffix "_expB" --pooling mean


echo "--- REUTERS: EPOCH 4 ---"
# Exp A (DoRA)
echo "[Reuters] Exp A (DoRA) - 4 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset reuters --epochs 4 --batch_size 8 --use_dora --lr_scheduler cosine
mv -f "$ADAPTER_ROOT/${MODEL}_reuters_4ep" "$ADAPTER_ROOT/${MODEL}_reuters_expA_4ep"
python -u main.py --model $MODEL --dataset reuters --lora --epochs 4 --suffix "_expA" --pooling mean

# Exp B (Dropout)
echo "[Reuters] Exp B (Dropout) - 4 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset reuters --epochs 4 --batch_size 16 --lora_dropout 0.3 --r 16 --lora_alpha 32
mv -f "$ADAPTER_ROOT/${MODEL}_reuters_4ep" "$ADAPTER_ROOT/${MODEL}_reuters_expB_4ep"
python -u main.py --model $MODEL --dataset reuters --lora --epochs 4 --suffix "_expB" --pooling mean


echo "--- REUTERS: EPOCH 5 ---"
# Exp A (DoRA)
echo "[Reuters] Exp A (DoRA) - 5 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset reuters --epochs 5 --batch_size 8 --use_dora --lr_scheduler cosine
mv -f "$ADAPTER_ROOT/${MODEL}_reuters_5ep" "$ADAPTER_ROOT/${MODEL}_reuters_expA_5ep"
python -u main.py --model $MODEL --dataset reuters --lora --epochs 5 --suffix "_expA" --pooling mean

# Exp B (Dropout)
echo "[Reuters] Exp B (Dropout) - 5 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset reuters --epochs 5 --batch_size 16 --lora_dropout 0.3 --r 16 --lora_alpha 32
mv -f "$ADAPTER_ROOT/${MODEL}_reuters_5ep" "$ADAPTER_ROOT/${MODEL}_reuters_expB_5ep"
python -u main.py --model $MODEL --dataset reuters --lora --epochs 5 --suffix "_expB" --pooling mean


# ========================================================
# 2. DARKREDDIT
# ========================================================

echo "--- DARKREDDIT: EPOCH 2 ---"
# Exp A
echo "[DarkReddit] Exp A (DoRA) - 2 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset darkreddit --epochs 2 --batch_size 8 --use_dora --lr_scheduler cosine
mv -f "$ADAPTER_ROOT/${MODEL}_darkreddit_2ep" "$ADAPTER_ROOT/${MODEL}_darkreddit_expA_2ep"
python -u main.py --model $MODEL --dataset darkreddit --lora --epochs 2 --suffix "_expA" --pooling mean

# Exp B
echo "[DarkReddit] Exp B (Dropout) - 2 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset darkreddit --epochs 2 --batch_size 16 --lora_dropout 0.3 --r 16 --lora_alpha 32
mv -f "$ADAPTER_ROOT/${MODEL}_darkreddit_2ep" "$ADAPTER_ROOT/${MODEL}_darkreddit_expB_2ep"
python -u main.py --model $MODEL --dataset darkreddit --lora --epochs 2 --suffix "_expB" --pooling mean


echo "--- DARKREDDIT: EPOCH 3 ---"
# Exp A
echo "[DarkReddit] Exp A (DoRA) - 3 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset darkreddit --epochs 3 --batch_size 8 --use_dora --lr_scheduler cosine
mv -f "$ADAPTER_ROOT/${MODEL}_darkreddit_3ep" "$ADAPTER_ROOT/${MODEL}_darkreddit_expA_3ep"
python -u main.py --model $MODEL --dataset darkreddit --lora --epochs 3 --suffix "_expA" --pooling mean

# Exp B
echo "[DarkReddit] Exp B (Dropout) - 3 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset darkreddit --epochs 3 --batch_size 16 --lora_dropout 0.3 --r 16 --lora_alpha 32
mv -f "$ADAPTER_ROOT/${MODEL}_darkreddit_3ep" "$ADAPTER_ROOT/${MODEL}_darkreddit_expB_3ep"
python -u main.py --model $MODEL --dataset darkreddit --lora --epochs 3 --suffix "_expB" --pooling mean


echo "--- DARKREDDIT: EPOCH 4 ---"
# Exp A
echo "[DarkReddit] Exp A (DoRA) - 4 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset darkreddit --epochs 4 --batch_size 8 --use_dora --lr_scheduler cosine
mv -f "$ADAPTER_ROOT/${MODEL}_darkreddit_4ep" "$ADAPTER_ROOT/${MODEL}_darkreddit_expA_4ep"
python -u main.py --model $MODEL --dataset darkreddit --lora --epochs 4 --suffix "_expA" --pooling mean

# Exp B
echo "[DarkReddit] Exp B (Dropout) - 4 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset darkreddit --epochs 4 --batch_size 16 --lora_dropout 0.3 --r 16 --lora_alpha 32
mv -f "$ADAPTER_ROOT/${MODEL}_darkreddit_4ep" "$ADAPTER_ROOT/${MODEL}_darkreddit_expB_4ep"
python -u main.py --model $MODEL --dataset darkreddit --lora --epochs 4 --suffix "_expB" --pooling mean


echo "--- DARKREDDIT: EPOCH 5 ---"
# Exp A
echo "[DarkReddit] Exp A (DoRA) - 5 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset darkreddit --epochs 5 --batch_size 8 --use_dora --lr_scheduler cosine
mv -f "$ADAPTER_ROOT/${MODEL}_darkreddit_5ep" "$ADAPTER_ROOT/${MODEL}_darkreddit_expA_5ep"
python -u main.py --model $MODEL --dataset darkreddit --lora --epochs 5 --suffix "_expA" --pooling mean

# Exp B
echo "[DarkReddit] Exp B (Dropout) - 5 Epochs..."
python -u src/train_lora.py --model $MODEL --dataset darkreddit --epochs 5 --batch_size 16 --lora_dropout 0.3 --r 16 --lora_alpha 32
mv -f "$ADAPTER_ROOT/${MODEL}_darkreddit_5ep" "$ADAPTER_ROOT/${MODEL}_darkreddit_expB_5ep"
python -u main.py --model $MODEL --dataset darkreddit --lora --epochs 5 --suffix "_expB" --pooling mean


echo "=== MARATHON COMPLETE ==="