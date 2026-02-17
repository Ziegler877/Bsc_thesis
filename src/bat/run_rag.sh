#!/bin/bash
#SBATCH -J RAG_Forensics
#SBATCH -A p71186
#SBATCH -t 04:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=results/logs/RAG_%j.out
#SBATCH --error=results/logs/RAG_%j.err

# 1. Load Environment
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411

PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "=================================================="
echo "   STARTING RAG PIPELINE (Prompting)"
echo "   Model: Llama 4 Scout"
echo "   Date: $(date)"
echo "=================================================="

# Define Retrieval Source (Your best embeddings)
# We use the '5 Epochs' version of E5 Large because it should be the smartest "Search Dog"
EMB_REUTERS="results/embeddings/e5_large_reuters_lora_5ep.pt"
EMB_REDDIT="results/embeddings/e5_large_darkreddit_lora_5ep.pt"

# --- RUN 1: REUTERS ---
# We use k=10 candidates because Scout has a huge context window
echo ""
echo ">>> [1/2] Running RAG on Reuters..."
python -u run_rag.py \
    --model scout \
    --dataset reuters \
    --embeddings $EMB_REUTERS \
    --k 10 \
    --limit 2500

# --- RUN 2: DARK REDDIT ---
echo ""
echo ">>> [2/2] Running RAG on DarkReddit..."
python -u run_rag.py \
    --model scout \
    --dataset darkreddit \
    --embeddings $EMB_REDDIT \
    --k 10 \
    --limit 2500

echo ""
echo "=== RAG FINISHED ==="
echo "Results saved in results/rag_results/"