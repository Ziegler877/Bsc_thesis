import os
import sys
import re
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, top_k_accuracy_score

# Hardcoded path to your exact local directory
TARGET_DIR = r"C:\Users\maxid\Universität\12._Semester\BSC\Programm\results\embeddings\dynamic"

MODEL_ORDER = ["SOTA", "E5-Small", "E5-Large", "Llama-2", "Llama-3"]

# Aligned SOTA Data rows with the new comprehensive layout
SOTA_DATA = {
    "reuters": [
        {"Model": "SOTA", "Variant": "SVM (N-Gram)",
         "C-Acc": 0.9234, "C-F1": 0.9230, "C-T3": "-", "C-T5": "-", "C-MRR": "-",
         "R-Top1": "-", "R-F1": "-", "R-Top3": "-", "R-Top5": "-", "R-MAP": "-", "LogLoss": "-"},
        {"Model": "SOTA", "Variant": "BERT-AA",
         "C-Acc": 0.8720, "C-F1": 0.8710, "C-T3": "-", "C-T5": "-", "C-MRR": "-",
         "R-Top1": "-", "R-F1": "-", "R-Top3": "-", "R-Top5": "-", "R-MAP": "-", "LogLoss": "-"}
    ],
    "darkreddit": [
        {"Model": "SOTA", "Variant": "LUAR (SOTA)",
         "C-Acc": 0.8200, "C-F1": "-", "C-T3": "-", "C-T5": "-", "C-MRR": "-",
         "R-Top1": "-", "R-F1": "-", "R-Top3": 0.9410, "R-Top5": "-", "R-MAP": "-", "LogLoss": "-"},
        {"Model": "SOTA", "Variant": "VeriDark (BERT)",
         "C-Acc": 0.6500, "C-F1": 0.6400, "C-T3": "-", "C-T5": "-", "C-MRR": "-",
         "R-Top1": "-", "R-F1": "-", "R-Top3": "-", "R-Top5": "-", "R-MAP": "-", "LogLoss": "-"}
    ]
}

FILE_MAPS = {
    "e5_small": "E5-Small",
    "e5_large": "E5-Large",
    "llama2": "Llama-2",
    "llama3": "Llama-3"
}


def parse_file_details(filename):
    """Parses exact configuration properties from filename."""
    fname = filename.lower()

    # Identify Model Core
    model = "Unknown"
    for pattern, name in FILE_MAPS.items():
        if pattern in fname:
            model = name
            break

    # Determine execution paradigm properties
    is_lora = "lora" in fname
    pooling = "Dynamic" if "dynamic" in fname else ("GeM" if "gmp" in fname else "Mean")
    chunking = "Chunked" if "chunked" in fname else "Truncated"
    is_sub5 = "sub5" in fname

    # Isolate LoRA run identifier
    run_match = re.search(r'run(\d+)', fname)
    run_id = f"Run {run_match.group(1)}" if run_match else None

    # Create unified base variation descriptor string
    sub_str = ", Sub-5" if is_sub5 else ""
    base_variant = f"{pooling}, {chunking}{sub_str}"

    return model, is_lora, base_variant, run_id


def evaluate_pt_file(file_path):
    """Loads vectors and calculates all 11 evaluation metrics."""
    try:
        data = torch.load(file_path, map_location="cpu")
        train_vecs = data.get('train_vecs', data.get('train_embeddings'))
        train_labels = data.get('train_labels')
        test_vecs = data.get('test_vecs', data.get('test_embeddings'))
        test_labels = data.get('test_labels')

        if train_vecs is None or test_vecs is None: return None

        # Clean vectors and convert to numpy arrays
        train_vecs = torch.nan_to_num(train_vecs, nan=0.0)
        test_vecs = torch.nan_to_num(test_vecs, nan=0.0)

        # Enforce unit normalization for accurate spatial dot products
        train_vecs = F.normalize(train_vecs, p=2, dim=1)
        test_vecs = F.normalize(test_vecs, p=2, dim=1)

        unique_classes = sorted(list(set(train_labels)))
        label_to_index = {name: i for i, name in enumerate(unique_classes)}
        n_classes = len(unique_classes)

        y_true = [label_to_index[l] for l in test_labels if l in label_to_index]
        if not y_true: return None

        # ----------------------------------------------------
        # METHODOLOGY 1: CENTROID BASE CALCULATIONS
        # ----------------------------------------------------
        centroids = []
        for label in unique_classes:
            indices = [i for i, x in enumerate(train_labels) if x == label]
            if not indices:
                centroids.append(torch.zeros(train_vecs.shape[1]))
            else:
                centroid = F.normalize(torch.mean(train_vecs[indices], dim=0), p=2, dim=0)
                centroids.append(centroid)

        centroid_matrix = torch.stack(centroids)
        c_scores = torch.mm(test_vecs, centroid_matrix.transpose(0, 1)).numpy()
        c_probs = F.softmax(torch.tensor(c_scores) * 10, dim=1).numpy()
        c_preds = np.argmax(c_scores, axis=1)

        c_acc = accuracy_score(y_true, c_preds)
        c_f1 = f1_score(y_true, c_preds, average='macro')
        c_t3 = top_k_accuracy_score(y_true, c_scores, k=min(3, n_classes))
        c_t5 = top_k_accuracy_score(y_true, c_scores, k=min(5, n_classes))

        # Centroid Reciprocal Rank Calculation
        c_ranked = np.argsort(-c_scores, axis=1)
        c_rr = [1.0 / (np.where(c_ranked[i] == y_true[i])[0][0] + 1) for i in range(len(y_true))]
        c_mrr = np.mean(c_rr)

        try:
            ll = log_loss(y_true, c_probs, labels=list(range(n_classes)))
        except:
            ll = -1.0

        # ----------------------------------------------------
        # METHODOLOGY 2: DOCUMENT-TO-DOCUMENT RETRIEVAL
        # ----------------------------------------------------
        sim_matrix = torch.mm(test_vecs, train_vecs.transpose(0, 1)).numpy()
        train_labels_np = np.array(train_labels)

        ap_scores = []
        r_preds = []
        r_top3_hits, r_top5_hits = 0, 0

        for i, query_label in enumerate(test_labels):
            query_sims = sim_matrix[i]
            sorted_idx = np.argsort(-query_sims)
            sorted_labels = train_labels_np[sorted_idx]

            # Top-1 Document Retrieval Selection
            r_preds.append(label_to_index.get(sorted_labels[0], 0))

            is_relevant = (sorted_labels == query_label)
            total_relevant = is_relevant.sum()

            if total_relevant == 0:
                ap_scores.append(0.0)
                continue

            # Soft Top-K Document Matching Counts
            if is_relevant[:3].any(): r_top3_hits += 1
            if is_relevant[:5].any(): r_top5_hits += 1

            # Exact Average Precision Equation execution
            hit_ranks = np.where(is_relevant)[0] + 1
            num_hits_at_rank = np.arange(1, len(hit_ranks) + 1)
            ap = (num_hits_at_rank / hit_ranks).sum() / total_relevant
            ap_scores.append(ap)

        r_top1 = accuracy_score(y_true, r_preds)
        r_f1 = f1_score(y_true, r_preds, average='macro')
        r_top3 = r_top3_hits / len(y_true)
        r_top5 = r_top5_hits / len(y_true)
        r_map = np.mean(ap_scores)

        return {
            "C-Acc": c_acc, "C-F1": c_f1, "C-T3": c_t3, "C-T5": c_t5, "C-MRR": c_mrr,
            "R-Top1": r_top1, "R-F1": r_f1, "R-Top3": r_top3, "R-Top5": r_top5, "R-MAP": r_map,
            "LogLoss": ll
        }
    except Exception as e:
        print(f"(!) Error reading file: {e}")
        return None


def main():
    if not os.path.exists(TARGET_DIR):
        print(f"(!) Specified directory does not exist: {TARGET_DIR}")
        return

    all_files = [f for f in os.listdir(TARGET_DIR) if f.endswith(".pt")]
    print(f"\n--- ULTIMATE COMPREHENSIVE EVALUATOR ---")
    print(f"Found {len(all_files)} embedding data files inside:\n{TARGET_DIR}\n")

    for dataset_name in ["reuters", "darkreddit"]:
        print(
            f"\n========================================= {dataset_name.upper()} DATASET =========================================")
        final_rows = []

        # Step 1: Inject Reference State-of-the-Art Rows
        if dataset_name in SOTA_DATA:
            for sota_row in SOTA_DATA[dataset_name]:
                final_rows.append(sota_row)

        # Process each model explicitly in order
        for model in MODEL_ORDER:
            if model == "SOTA": continue

            # Gather matching files for this specific dataset and model core
            model_files = [f for f in all_files if dataset_name in f.lower() and
                           any(p in f.lower() for p in [k for k, v in FILE_MAPS.items() if v == model])]

            # Step 2: Separate into Baseline (Non-LoRA) and Fine-tuned (LoRA) Arrays
            non_lora_records = []
            lora_groups = {}  # Dict grouping structural configs: { base_variant: [run_metrics_dicts] }

            for f in model_files:
                f_model, is_lora, base_variant, run_id = parse_file_details(f)
                if f_model != model: continue

                metrics = evaluate_pt_file(os.path.join(TARGET_DIR, f))
                if not metrics: continue

                record = {"Model": model, "Variant": f"Base ({base_variant})"}
                record.update(metrics)

                if not is_lora:
                    non_lora_records.append(record)
                else:
                    if base_variant not in lora_groups: lora_groups[base_variant] = []
                    record["Variant"] = f"LoRA ({base_variant}) - {run_id if run_id else 'Run'}"
                    lora_groups[base_variant].append(record)

            # Append all baseline rows first
            final_rows.extend(non_lora_records)

            # Step 3: Process LoRA rows followed immediately by their Averages
            for base_variant, runs in lora_groups.items():
                # Sort runs sequentially (Run 1, Run 2, etc.)
                runs = sorted(runs, key=lambda x: x["Variant"])
                final_rows.extend(runs)

                # Compute mathematical averages across columns for these specific runs
                metric_keys = ["C-Acc", "C-F1", "C-T3", "C-T5", "C-MRR", "R-Top1", "R-F1", "R-Top3", "R-Top5", "R-MAP",
                               "LogLoss"]
                avg_record = {"Model": model, "Variant": f"LoRA ({base_variant}) - Avg"}

                for k in metric_keys:
                    avg_record[k] = np.mean([r[k] for r in runs])

                final_rows.append(avg_record)

        if not final_rows: continue
        df = pd.DataFrame(final_rows)

        # Print Consolidated Layout
        print("-" * 155)
        print(
            f"{'Model':<12} {'Variant / Configuration Structure':<32} | {'C-Acc':<6} {'C-F1':<6} {'C-T3':<6} {'C-T5':<6} {'C-MRR':<6} | {'R-Top1':<6} {'R-F1':<6} {'R-Top3':<6} {'R-Top5':<6} {'R-MAP':<6} | {'LogLoss':<6}")
        print("-" * 155)
        for _, row in df.iterrows():
            def fmt(val): return f"{float(val):.3f}" if (val != "-" and not pd.isna(val)) else "-"

            # Highlight Avg rows visually to track scaling
            v_name = str(row['Variant'])
            spacer = ">>> " if "- Avg" in v_name else "  "
            print(
                f"{str(row['Model']):<12} {spacer + v_name:<32} | {fmt(row['C-Acc']):<6} {fmt(row['C-F1']):<6} {fmt(row['C-T3']):<6} {fmt(row['C-T5']):<6} {fmt(row['C-MRR']):<6} | {fmt(row['R-Top1']):<6} {fmt(row['R-F1']):<6} {fmt(row['R-Top3']):<6} {fmt(row['R-Top5']):<6} {fmt(row['R-MAP']):<6} | {fmt(row['LogLoss']):<6}")
        print("-" * 155 + "\n")

        # Save Output CSV File
        save_path = os.path.join(TARGET_DIR, f"comprehensive_results_{dataset_name}.csv")
        df.to_csv(save_path, index=False)
        print(f"   [Saved Data] CSV written successfully: {save_path}\n")


if __name__ == "__main__":
    main()