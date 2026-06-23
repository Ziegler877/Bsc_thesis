import os
import sys
import torch

# 1. ENVIRONMENT DETECTION & ROOT PATHS
IS_CLUSTER = (os.name != 'nt')

if IS_CLUSTER:
    print("   [Config] Detected CLUSTER Environment (Linux).")

    # 1. Root Directory (Where config.py is located)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # 2. Big Storage Paths (Cluster)
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

    # 3. Model Checkpoints
    LLAMA2_CHECKPOINT_DIR = os.path.join(MODELS_DIR, "Llama-2-7b-hf")
    LLAMA4_CHECKPOINT_DIR = os.path.join(MODELS_DIR, "Llama-4-Maverick")

    # NEW: Llama 3.1
    LLAMA3_CHECKPOINT_DIR = os.path.join(MODELS_DIR, "Llama-3.1-8B")

    # E5 Models
    # Updated to match your 'ls' output
    E5_SMALL_ID = os.path.join(MODELS_DIR, "e5-small")
    E5_LARGE_ID = os.path.join(MODELS_DIR, "e5-large")

else:
    print("   [Config] Detected LOCAL Environment (Windows).")

    # 1. Root Directory
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

    # 2. Local Model Paths
    # Root for models: D:\BSC_Thesis_Models

    LLAMA2_CHECKPOINT_DIR = r"D:\BSC_Thesis_Models\Llama-2-7b-hf"
    LLAMA3_CHECKPOINT_DIR = "meta-llama/Meta-Llama-3.1-8B"

    # E5 Models (Local Paths)
    E5_SMALL_ID = r"D:\12._Semester\BSC\Cluster\Models\e5-small"
    E5_LARGE_ID = r"D:\12._Semester\BSC\Cluster\Models\e5-large"

# 2. OUTPUT SUB-FOLDERS
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
EMBEDDINGS_DIR = os.path.join(RESULTS_DIR, "embeddings")
ADAPTERS_DIR = os.path.join(RESULTS_DIR, "adapters")
RAG_DIR = os.path.join(RESULTS_DIR, "rag_results")

# Create them if they don't exist
for d in [LOGS_DIR, PLOTS_DIR, EMBEDDINGS_DIR, ADAPTERS_DIR, RAG_DIR]:
    os.makedirs(d, exist_ok=True)

# 3. DATASET PATHS

# --- REUTERS ---
RAW_REUTERS_ROOT = os.path.join(DATA_DIR, "raw", "reuter+50+50")
REUTERS_TRAIN_DIR = os.path.join(RAW_REUTERS_ROOT, "C50train")
REUTERS_TEST_DIR = os.path.join(RAW_REUTERS_ROOT, "C50test")
REUTERS_VAL_DIR = os.path.join(RAW_REUTERS_ROOT, "C50val")

# --- DARK REDDIT ---
RAW_DARKREDDIT_ROOT = os.path.join(DATA_DIR, "raw", "darkreddit_authorship_attribution_anon")
DARK_REDDIT_TRAIN = os.path.join(RAW_DARKREDDIT_ROOT, "darkreddit_authorship_attribution_train_anon.jsonl")
DARK_REDDIT_TEST = os.path.join(RAW_DARKREDDIT_ROOT, "darkreddit_authorship_attribution_test_anon.jsonl")
DARK_REDDIT_VAL = os.path.join(RAW_DARKREDDIT_ROOT, "darkreddit_authorship_attribution_val_anon.jsonl")

# 4. HARDWARE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    print(f"--- CONFIGURATION CHECK ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"E5 Small Path: {E5_SMALL_ID}")
    print(f"Llama 3 Path:  {LLAMA3_CHECKPOINT_DIR}")