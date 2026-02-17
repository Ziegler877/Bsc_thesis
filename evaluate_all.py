import os
import sys

# --- FIX: Add parent directory to path so we can find 'config.py' ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
# --------------------------------------------------------------------

import config

# Standard library imports
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ==========================================
#  CONFIGURATION & SOTA DATA
# ==========================================

# 1. Define the exact order you want rows to appear
MODEL_ORDER = [
    "SOTA ",
    "E5-Small",
    "E5-Large",
    "Llama-2",
    "Llama-4-Scout"  # <--- Changed from Generic Llama-4 to Scout
]

# 2. Hardcoded SOTA Benchmarks (for comparison)
SOTA_DATA = {
    "reuters": [
        {"Model": "SOTA", "Variant": "SVM (N-Gram)", "Accuracy": 0.9234, "F1 Macro": 0.92, "Precision": "-",
         "Recall": "-"},
        {"Model": "SOTA", "Variant": "BERT-AA", "Accuracy": 0.8720, "F1 Macro": 0.87, "Precision": "-", "Recall": "-"},
    ],
    "darkreddit": [
        {"Model": "SOTA", "Variant": "LUAR (SOTA)", "Accuracy": 0.8200, "F1 Macro": 0.82, "Precision": "-",
         "Recall": "-"},
        {"Model": "SOTA", "Variant": "VeriDark (BERT)", "Accuracy": 0.6500, "F1 Macro": 0.64, "Precision": "-",
         "Recall": "-"},
    ]
}

# 3. References for the Footer
SOTA_REFS = {
    "SVM (N-Gram)": "Houvardas & Stamatatos (2006) - 'N-gram Feature Selection for Authorship Attribution'",
    "BERT-AA": "Fabien et al. (2020) - 'BERTAA: BERT-based Authorship Attribution' (https://arxiv.org/abs/2009.07722)",
    "LUAR (SOTA)": "Rivera-Soto et al. (2021) - 'Learning Universal Authorship Representations' (EMNLP)",
    "VeriDark (BERT)": "He et al. (2023) - 'VeriDark: Authorship Verification on the Dark Web'"
}

# 4. Search patterns for your files
# These strings are matched against the filename (case-insensitive)
FILE_PATTERNS = {
    "E5-Small": ["e5_small"],
    "E5-Large": ["e5_large"],
    "Llama-2": ["llama2", "llama_2"],
    "Llama-4-Scout": ["scout", "llama4_scout"]  # <--- Looks for your new files
}


# ==========================================
#  HELPER FUNCTIONS
# ==========================================

def parse_epochs(filename):
    """Tries to find 'ep3', 'ep5' or '_base' in the filename."""
    filename = filename.lower()

    if "ep3" in filename or "_3ep" in filename:
        return "3 Epochs"
    elif "ep5" in filename or "_5ep" in filename:
        return "5 Epochs"
    elif "base" in filename:
        return "Base (No LoRA)"
    elif "lora" in filename:
        return "LoRA (Unk Ep)"
    return "Base"


def evaluate_embeddings(file_path):
    """Loads a .pt file and calculates metrics."""
    try:
        # --- FIX: weights_only=False stops the FutureWarning ---
        data = torch.load(file_path, map_location="cpu", weights_only=False)

        # Robust loading of keys
        train_vecs = data.get('train_vecs', data.get('train_embeddings'))
        train_labels = data.get('train_labels')
        test_vecs = data.get('test_vecs', data.get('test_embeddings'))
        test_labels = data.get('test_labels')

        if train_vecs is None or test_vecs is None:
            return None

        # --- CLASSIFICATION LOGIC (Centroids) ---
        unique_classes = sorted(list(set(train_labels)))
        label_to_index = {name: i for i, name in enumerate(unique_classes)}

        centroids = []
        for label in unique_classes:
            indices = [i for i, x in enumerate(train_labels) if x == label]
            if not indices:
                centroids.append(torch.zeros(train_vecs.shape[1]))
            else:
                indices_tensor = torch.tensor(indices)
                class_vecs = train_vecs.index_select(0, indices_tensor)
                centroid = F.normalize(torch.mean(class_vecs, dim=0), p=2, dim=0)
                centroids.append(centroid)

        centroid_matrix = torch.stack(centroids)

        # Predict
        scores = torch.mm(test_vecs, centroid_matrix.transpose(0, 1))
        predictions = torch.argmax(scores, dim=1)

        # Filter valid
        y_true = []
        y_pred = []
        for i, label in enumerate(test_labels):
            if label in label_to_index:
                y_true.append(label_to_index[label])
                y_pred.append(predictions[i])

        if not y_true:
            return None

        # Calc Metrics
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1 Macro": f1_score(y_true, y_pred, average='macro'),
            "Precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "Recall": recall_score(y_true, y_pred, average='macro', zero_division=0)
        }

    except Exception as e:
        print(f"(!) Error reading {os.path.basename(file_path)}: {e}")
        return None


# ==========================================
#  MAIN LOGIC
# ==========================================

def main():
    print(f"--- EVALUATION SUMMARY ---")

    # --- FIX: Use EMBEDDINGS_DIR instead of PROCESSED_DIR ---
    target_dir = config.EMBEDDINGS_DIR

    print(f"Source Directory: {target_dir}")

    # Check dir
    if not os.path.exists(target_dir):
        print(f"(!) Directory not found: {target_dir}")
        return

    all_files = [f for f in os.listdir(target_dir) if f.endswith(".pt")]

    # PROCESS EACH DATASET SEPARATELY
    for dataset_name in ["reuters", "darkreddit"]:
        print(f"\n" + "=" * 80)
        print(f"   RESULTS TABLE: {dataset_name.upper()}")
        print("=" * 80)

        table_rows = []

        # 1. ADD SOTA ROWS
        if dataset_name in SOTA_DATA:
            for sota_row in SOTA_DATA[dataset_name]:
                table_rows.append(sota_row)

        # 2. SCAN FOR OUR MODELS
        for model_name in MODEL_ORDER:
            if "SOTA" in model_name: continue

            search_terms = FILE_PATTERNS.get(model_name, [])
            matches = [f for f in all_files if dataset_name in f.lower() and any(t in f.lower() for t in search_terms)]

            if not matches:
                table_rows.append({
                    "Model": model_name, "Variant": "-",
                    "Accuracy": "-", "F1 Macro": "-", "Precision": "-", "Recall": "-"
                })
                continue

            for f in matches:
                full_path = os.path.join(target_dir, f)
                metrics = evaluate_embeddings(full_path)

                if metrics:
                    variant = parse_epochs(f)
                    row = {
                        "Model": model_name,
                        "Variant": variant,
                        "Accuracy": metrics["Accuracy"],
                        "F1 Macro": metrics["F1 Macro"],
                        "Precision": metrics["Precision"],
                        "Recall": metrics["Recall"]
                    }
                    table_rows.append(row)

        # 3. BUILD DATAFRAME
        df = pd.DataFrame(table_rows)

        # Convert cols to string for formatted printing
        format_cols = ["Accuracy", "F1 Macro", "Precision", "Recall"]
        for col in format_cols:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x)

        # 4. CUSTOM PRINTING (With Lines between Groups)
        # Groups: SOTA, E5, Llama

        # Header
        header = f"{'Model':<15} {'Variant':<20} {'Accuracy':<10} {'F1 Macro':<10} {'Precision':<10} {'Recall':<10}"
        print(header)
        print("-" * 80)

        current_group = "SOTA"  # Start assuming SOTA

        # We manually iterate to control the lines
        for _, row in df.iterrows():
            m_name = str(row['Model'])

            # Detect Group Change
            new_group = "Other"
            if "SOTA" in m_name:
                new_group = "SOTA"
            elif "E5" in m_name:
                new_group = "E5"
            elif "Llama" in m_name:
                new_group = "Llama"

            # If group changed (and it's not the very first line), print separator
            if new_group != current_group:
                print("-" * 80)
                current_group = new_group

            # Print Row
            print(
                f"{str(row['Model']):<15} {str(row['Variant']):<20} {str(row['Accuracy']):<10} {str(row['F1 Macro']):<10} {str(row['Precision']):<10} {str(row['Recall']):<10}")

        print("-" * 80)

        # 5. PRINT REFERENCES FOOTER
        print("\n[References]")
        if dataset_name in SOTA_DATA:
            used_sota = SOTA_DATA[dataset_name]
            for item in used_sota:
                variant = item['Variant']
                if variant in SOTA_REFS:
                    print(f"* {variant}: {SOTA_REFS[variant]}")

        # 6. SAVE TO CSV
        save_path = os.path.join(target_dir, f"final_results_{dataset_name}.csv")
        df.to_csv(save_path, index=False)
        print(f"\nSaved CSV to: {save_path}")


if __name__ == "__main__":
    main()