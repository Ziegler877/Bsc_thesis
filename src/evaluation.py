import os
import sys
import datetime
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score
)

# Link to config
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.append(project_root)
import config


def ensure_directories():
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)


def calculate_centroids(train_vecs, train_labels, unique_authors, device):
    """
    Calculates normalized centroids (Prototypes) for each author.
    """
    centroids = []
    for author in unique_authors:
        indices = [i for i, x in enumerate(train_labels) if x == author]
        if not indices:
            centroid = torch.zeros(train_vecs.shape[1]).to(device)
        else:
            indices_tensor = torch.tensor(indices).to(device)
            author_vecs = train_vecs.index_select(0, indices_tensor)

            # Standard Mean Pooling for Centroid
            centroid = torch.mean(author_vecs, dim=0)

            # --- FIX: Handle potential NaNs/Infs before Normalization ---
            centroid = torch.nan_to_num(centroid, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalization
            centroid = F.normalize(centroid, p=2, dim=0)

        centroids.append(centroid)
    return torch.stack(centroids)


def run_evaluation(
        train_vecs,
        test_vecs,
        train_labels,
        test_labels,
        model_name,
        dataset_name,
        device="cuda",
        extra_info="",
        pooling="mean",
        chunking=False
):
    ensure_directories()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mode_str = "CHUNKED" if chunking else "TRUNCATED"

    print(f"   [Eval] Calculating metrics for {model_name} (Pool: {pooling}, Mode: {mode_str}) on {dataset_name}...")

    # --- FIX: Sanitize Inputs immediately ---
    if torch.isnan(train_vecs).any() or torch.isinf(train_vecs).any():
        print("   [Warn] Train vectors contain NaNs/Infs! Cleaning...")
        train_vecs = torch.nan_to_num(train_vecs, nan=0.0)

    if torch.isnan(test_vecs).any() or torch.isinf(test_vecs).any():
        print("   [Warn] Test vectors contain NaNs/Infs! Cleaning...")
        test_vecs = torch.nan_to_num(test_vecs, nan=0.0)
    # ----------------------------------------

    # Setup
    unique_authors = sorted(list(set(train_labels)))
    label_to_index = {name: i for i, name in enumerate(unique_authors)}

    valid_indices = [i for i, l in enumerate(test_labels) if l in label_to_index]
    if len(valid_indices) < len(test_labels):
        print(f"   [Eval] Warning: Dropping {len(test_labels) - len(valid_indices)} test samples with unseen labels.")
        test_vecs = test_vecs[valid_indices]
        test_labels = [test_labels[i] for i in valid_indices]

    true_indices = [label_to_index[l] for l in test_labels]

    train_vecs = train_vecs.to(device)
    test_vecs = test_vecs.to(device)

    # 1. Create Author Prototypes
    centroid_matrix = calculate_centroids(train_vecs, train_labels, unique_authors, device)

    # Comparison
    similarity_matrix = torch.mm(test_vecs, centroid_matrix.transpose(0, 1))

    # Scaling
    probs = F.softmax(similarity_matrix * 10, dim=1).cpu().numpy()
    pred_indices = torch.argmax(similarity_matrix, dim=1).cpu().numpy()

    # Metrics
    try:
        acc_top1 = accuracy_score(true_indices, pred_indices)

        k = 3 if len(unique_authors) >= 3 else len(unique_authors)
        # Check for NaNs in similarity matrix before Top-K
        sim_cpu = similarity_matrix.cpu().numpy()
        sim_cpu = np.nan_to_num(sim_cpu)
        acc_top3 = top_k_accuracy_score(true_indices, sim_cpu, k=k)

        f1_macro = f1_score(true_indices, pred_indices, average='macro')
        f1_weighted = f1_score(true_indices, pred_indices, average='weighted')

        # Robust Log Loss
        probs = np.nan_to_num(probs, nan=1.0 / len(unique_authors))
        ll = log_loss(true_indices, probs, labels=list(range(len(unique_authors))))

        report_str = classification_report(true_indices, pred_indices, target_names=unique_authors)

        print(f"   [Eval] Results -> Acc: {acc_top1:.4f} | F1: {f1_macro:.4f}")

    except Exception as e:
        print(f"   [Eval] CRITICAL ERROR during metric calculation: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Save Log
    chunk_tag = "chunked" if chunking else "truncated"
    base_filename = f"{model_name}_{dataset_name}_{pooling}_{chunk_tag}_{timestamp}"
    log_path = os.path.join(config.LOGS_DIR, f"{base_filename}_report.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"EVALUATION REPORT: {model_name} | {dataset_name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date:       {timestamp}\n")
        f.write(f"Pooling:    {pooling.upper()}\n")
        f.write(f"Chunking:   {chunking} ({mode_str})\n")
        f.write(f"Info:       {extra_info}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Top-1 Accuracy:   {acc_top1:.4f}\n")
        f.write(f"Top-3 Accuracy:   {acc_top3:.4f}\n")
        f.write(f"Macro F1-Score:   {f1_macro:.4f}\n")
        f.write(f"Log Loss:         {ll:.4f}\n")
        f.write("-" * 60 + "\n")
        f.write("DETAILED REPORT:\n")
        f.write(report_str)
        f.write("=" * 60 + "\n")
    print(f"   [Eval] Report saved: {log_path}")

    # Save Plot
    plot_path = os.path.join(config.PLOTS_DIR, f"{base_filename}_matrix.png")
    cm = confusion_matrix(true_indices, pred_indices)
    plt.figure(figsize=(20, 15))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=unique_authors, yticklabels=unique_authors)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    title_str = f'{model_name} | {dataset_name}\nPool: {pooling.upper()} | Mode: {mode_str}\nAcc: {acc_top1:.2f} | F1: {f1_macro:.2f}'
    plt.title(title_str)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {
        "test_accuracy": acc_top1,
        "test_top3_accuracy": acc_top3,
        "test_f1_macro": f1_macro,
        "test_f1_weighted": f1_weighted,
        "test_log_loss": ll
    }