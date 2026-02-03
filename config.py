import os
import sys
import torch

# =================================================================
# 1. ENVIRONMENT DETECTION & ROOT PATHS
# =================================================================

# Check if we are on the Cluster (Linux) or Laptop (Windows)
IS_CLUSTER = (os.name != 'nt')

if IS_CLUSTER:
    print("   [Config] Detected CLUSTER Environment (Linux).")

    # 1. Root Directory (Where config.py is located)
    # On Cluster: /gpfs/data/fs71186/ziegler/ThesisProject
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # 2. Big Storage Paths (Everything is inside ThesisProject now)
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

    # 3. Model Checkpoints
    LLAMA2_CHECKPOINT_DIR = os.path.join(MODELS_DIR, "Llama-2-7b-hf")
    LLAMA4_CHECKPOINT_DIR = os.path.join(MODELS_DIR, "Llama-4-Maverick")
    # NEW: Scout path for Cluster
    LLAMA4_SCOUT_CHECKPOINT_DIR = os.path.join(MODELS_DIR, "Llama-4-Scout")

else:
    print("   [Config] Detected LOCAL Environment (Windows).")

    # 1. Root Directory
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

    # 2. Local Model Paths (Update these if they change on your laptop)
    LLAMA2_CHECKPOINT_DIR = r"D:\.llama\checkpoints\Llama-2-7b-hf"
    LLAMA4_CHECKPOINT_DIR = r"D:\.llama\checkpoints\Llama-4-Maverick-17B-128E-Instruct"
    # Placeholder for Scout on Windows
    LLAMA4_SCOUT_CHECKPOINT_DIR = r"D:\.llama\checkpoints\Llama-4-Scout"

# =================================================================
# 2. OUTPUT SUB-FOLDERS
# =================================================================
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
EMBEDDINGS_DIR = os.path.join(RESULTS_DIR, "embeddings")
ADAPTERS_DIR = os.path.join(RESULTS_DIR, "adapters")
# --- NEW: Folder for RAG Results ---
RAG_DIR = os.path.join(RESULTS_DIR, "rag_results")

# Create them if they don't exist
for d in [LOGS_DIR, PLOTS_DIR, EMBEDDINGS_DIR, ADAPTERS_DIR, RAG_DIR]:
    os.makedirs(d, exist_ok=True)

# =================================================================
# 3. DATASET PATHS
# =================================================================

# --- REUTERS ---
RAW_REUTERS_ROOT = os.path.join(DATA_DIR, "raw", "reuter+50+50")
REUTERS_TRAIN_DIR = os.path.join(RAW_REUTERS_ROOT, "C50train")
REUTERS_TEST_DIR = os.path.join(RAW_REUTERS_ROOT, "C50test")

# --- DARK REDDIT ---
RAW_DARKREDDIT_ROOT = os.path.join(DATA_DIR, "raw", "darkreddit_authorship_attribution_anon")
DARK_REDDIT_TRAIN = os.path.join(RAW_DARKREDDIT_ROOT, "darkreddit_authorship_attribution_train_anon.jsonl")
DARK_REDDIT_TEST = os.path.join(RAW_DARKREDDIT_ROOT, "darkreddit_authorship_attribution_test_anon.jsonl")
DARK_REDDIT_VAL = os.path.join(RAW_DARKREDDIT_ROOT, "darkreddit_authorship_attribution_val_anon.jsonl")

# =================================================================
# 4. HARDWARE
# =================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    print(f"--- CONFIGURATION CHECK ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"RAG Results:  {RAG_DIR}")
    print(f"Llama Scout:  {LLAMA4_SCOUT_CHECKPOINT_DIR}")