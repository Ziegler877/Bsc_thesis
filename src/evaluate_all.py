import os
import sys
import math
import re  # Added for regex parsing

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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
    log_loss
)

# ==========================================
#  CONFIGURATION & SOTA DATA
# ==========================================

# Updated Model Order to include Llama-3
MODEL_ORDER = [
    "SOTA",
    "E5-Small",
    "E5-Large",
    "Llama-2",
    "Llama-3",
    "Llama-4-Scout"
]

# Hardcoded SOTA Benchmarks
SOTA_DATA = {
    "reuters": [
        {"Model": "SOTA", "Variant": "SVM (N-Gram)",
         "Accuracy": 0.9234, "Top-3": "-", "F1 Macro": 0.9230, "Log Loss": "-"},

        {"Model": "SOTA", "Variant": "BERT-AA",
         "Accuracy": 0.8720, "Top-3": "-", "F1 Macro": 0.8710, "Log Loss": "-"},
    ],
    "darkreddit": [
        {"Model": "SOTA", "Variant": "LUAR (SOTA)",
         "Accuracy": 0.8200, "Top-3": 0.9410, "F1 Macro": "-", "Log Loss": "-"},

        {"Model": "SOTA", "Variant": "VeriDark (BERT)",
         "Accuracy": 0.6500, "Top-3": "-", "F1 Macro": 0.6400, "Log Loss": "-"},
    ]
}

FILE_PATTERNS = {
    "E5-Small": ["e5_small"],
    "E5-Large": ["e5_large"],
    "Llama-2": ["llama2", "llama_2"],
    "Llama-3": ["llama3", "llama_3"],
    "Llama-4-Scout": ["scout", "llama4_scout"]
}


# ==========================================
#  HELPER FUNCTIONS
# ==========================================

def parse_variant(filename):
    """
    Parses filename to extract: Epochs, Pooling, Chunking, AND Subset.
    Format: "LoRA 3ep (GeM, Chunked, Sub-5)" or "Base (Mean, Trunc)"
    """
    filename = filename.lower()

    # 1. Determine Training Mode (Base vs LoRA)
    if "base" in filename:
        mode = "Base"
    elif "ep" in filename:
        # Try to extract exact epoch number
        match = re.search(r'_(\d+)ep', filename)
        if match:
            mode = f"LoRA {match.group(1)}ep"
        else:
            mode = "LoRA"
    else:
        mode = "Base"  # Default if unsure

    # 2. Determine Pooling
    if "_gmp" in filename:
        pooling = "GeM"
    else:
        pooling = "Mean"

    # 3. Determine Chunking
    if "_chunked" in filename:
        chunking = "Chunked"
    else:
        chunking = "Truncated"

    # 4. Determine Subset (New)
    subset_str = ""
    match_sub = re.search(r'_sub(\d+)', filename)
    if match_sub:
        subset_str = f", Sub-{match_sub.group(1)}"

    return f"{mode} ({pooling}, {chunking}{subset_str})"


def evaluate_embeddings(file_path):
    """Loads a .pt file and calculates ALL metrics."""
    try:
        # Load safely
        data = torch.load(file_path, map_location="cpu")

        train_vecs = data.get('train_vecs', data.get('train_embeddings'))
        train_labels = data.get('train_labels')
        test_vecs = data.get('test_vecs', data.get('test_embeddings'))
        test_labels = data.get('test_labels')

        if train_vecs is None or test_vecs is None:
            print(f"(!) Missing vectors in {os.path.basename(file_path)}")
            return None

        # --- CENTROID CLASSIFICATION LOGIC ---
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
        scores = torch.mm(test_vecs, centroid_matrix.transpose(0, 1))

        probs_all = F.softmax(scores * 10, dim=1)
        predictions_all = torch.argmax(scores, dim=1)

        y_true = []
        y_pred = []
        y_scores_filtered = []
        y_probs_filtered = []

        for i, label in enumerate(test_labels):
            if label in label_to_index:
                y_true.append(label_to_index[label])
                y_pred.append(predictions_all[i].item())
                y_scores_filtered.append(scores[i].numpy())
                y_probs_filtered.append(probs_all[i].numpy())

        if not y_true:
            print(f"(!) No matching labels in {os.path.basename(file_path)}")
            return None

        y_scores_filtered = np.array(y_scores_filtered)
        y_probs_filtered = np.array(y_probs_filtered)

        n_classes = len(unique_classes)
        k = 3 if n_classes >= 3 else n_classes

        try:
            ll = log_loss(y_true, y_probs_filtered, labels=list(range(n_classes)))
        except:
            ll = -1.0

        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Top-3": top_k_accuracy_score(y_true, y_scores_filtered, k=k),
            "F1 Macro": f1_score(y_true, y_pred, average='macro'),
            "F1 Wtd": f1_score(y_true, y_pred, average='weighted'),
            "Precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "Recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
            "Log Loss": ll
        }
    except Exception as e:
        print(f"(!) Error reading {os.path.basename(file_path)}: {e}")
        return None


# ==========================================
#  PLOTTING FUNCTIONS
# ==========================================

def prepare_plot_data(df):
    """Cleans dataframe for plotting (converts strings to floats, handles '-')"""
    plot_df = df.copy()
    numeric_cols = ["Accuracy", "Top-3", "F1 Macro", "F1 Wtd", "Precision", "Recall"]

    for col in numeric_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')

    # Create Display Name
    plot_df['Display Name'] = plot_df['Model'] + "\n" + plot_df['Variant']
    return plot_df


def plot_grouped_bar(df, dataset_name, output_dir):
    """1. Grouped Bar Chart: Accuracy Comparison"""
    plot_df = prepare_plot_data(df)
    plot_df = plot_df.dropna(subset=['Accuracy'])

    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")

    ax = sns.barplot(
        data=plot_df,
        x='Display Name',
        y='Accuracy',
        hue='Model',
        palette="viridis",
        dodge=False
    )

    plt.title(f"Leaderboard: {dataset_name.upper()}", fontsize=16)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"plot_1_bar_{dataset_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   [Plot] Saved Bar Chart: {save_path}")


def plot_radar_chart(df, dataset_name, output_dir):
    """2. Radar Chart: Compares E5-Small Variants"""
    plot_df = prepare_plot_data(df)

    # Filter: Only E5-Small and exclude SOTA
    target_models = plot_df[plot_df['Model'] == "E5-Small"]

    if len(target_models) < 2:
        print("   [Info] Not enough E5-Small variants for Radar Chart.")
        return

    # If too many, pick top 5 by Accuracy to avoid clutter
    if len(target_models) > 5:
        target_models = target_models.sort_values(by="Accuracy", ascending=False).head(5)

    metrics = ['Accuracy', 'F1 Macro', 'Recall', 'Precision']
    categories = metrics
    N = len(categories)

    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Dynamic color palette
    colors = sns.color_palette("husl", len(target_models))

    for i, (idx, row) in enumerate(target_models.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]

        # Clean label for legend
        label = row['Variant']

        ax.plot(angles, values, linewidth=2, linestyle='solid', label=label, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    plt.xticks(angles[:-1], categories)
    plt.ylim(0, 1.0)
    plt.title(f"Variant Impact: E5-Small ({dataset_name})", size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    save_path = os.path.join(output_dir, f"plot_2_radar_{dataset_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   [Plot] Saved Radar Chart: {save_path}")


def plot_heatmap(df, dataset_name, output_dir):
    """3. Heatmap: Overview of all metrics"""
    plot_df = prepare_plot_data(df)
    metrics = ["Accuracy", "F1 Macro", "Top-3", "Precision", "Recall"]

    heatmap_data = plot_df.drop_duplicates(subset=['Display Name']).set_index('Display Name')[metrics]
    heatmap_data = heatmap_data.astype(float)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        linewidths=.5,
        cbar_kws={'label': 'Score'}
    )

    plt.title(f"Metric Heatmap: {dataset_name.upper()}", fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"plot_3_heatmap_{dataset_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   [Plot] Saved Heatmap: {save_path}")


def plot_line_chart(df, dataset_name, output_dir):
    """4. Line Plot: Scaling Trend (Small -> Large -> Llama)"""
    plot_df = prepare_plot_data(df)

    # Size mapping
    size_map = {
        "E5-Small": 1,
        "E5-Large": 2,
        "Llama-2": 3,
        "Llama-3": 4,
        "Llama-4-Scout": 5
    }
    plot_df['Size_Rank'] = plot_df['Model'].map(size_map)
    plot_df = plot_df.dropna(subset=['Size_Rank', 'Accuracy'])

    # === AGGREGATION LOGIC ===
    # Take the BEST Performing variant for each model size to draw the trend line.
    best_variants = plot_df.loc[plot_df.groupby("Model")["Accuracy"].idxmax()]
    best_variants = best_variants.sort_values('Size_Rank')

    plt.figure(figsize=(10, 6))

    # Plot the Best Performing Trend Line
    sns.lineplot(
        data=best_variants,
        x='Model',
        y='Accuracy',
        marker='o',
        label='Best Variant Trend',
        linewidth=3,
        color='blue'
    )

    # Scatter all other points in background
    sns.scatterplot(
        data=plot_df,
        x='Model',
        y='Accuracy',
        hue='Variant',
        s=100,
        alpha=0.6
    )

    plt.title(f"Scaling Trend: Model Size vs Accuracy ({dataset_name})", fontsize=16)
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"plot_4_line_{dataset_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   [Plot] Saved Line Plot: {save_path}")


# ==========================================
#  MAIN LOGIC
# ==========================================

def main():
    print(f"--- EVALUATION & PLOTTING ---")

    local_dir = config.EMBEDDINGS_DIR
    cluster_dl_dir = os.path.join(config.RESULTS_DIR, "embeddings_downloaded")

    if os.path.exists(cluster_dl_dir) and len(os.listdir(cluster_dl_dir)) > 0:
        target_dir = cluster_dl_dir
        print(f"   [Source] Using DOWNLOADED embeddings from: {target_dir}")
    else:
        target_dir = local_dir
        print(f"   [Source] Using LOCAL/STANDARD embeddings from: {target_dir}")

    if not os.path.exists(target_dir):
        print(f"(!) Directory not found: {target_dir}")
        return

    all_files = [f for f in os.listdir(target_dir) if f.endswith(".pt")]
    print(f"   [Info] Found {len(all_files)} .pt files.")

    for dataset_name in ["reuters", "darkreddit"]:
        print(f"\nProcessing {dataset_name.upper()}...")
        table_rows = []

        # 1. Add SOTA
        if dataset_name in SOTA_DATA:
            for sota_row in SOTA_DATA[dataset_name]:
                row = sota_row.copy()
                for key in ["Accuracy", "Top-3", "F1 Macro", "F1 Wtd", "Precision", "Recall", "Log Loss"]:
                    if key not in row: row[key] = "-"
                table_rows.append(row)

        # 2. Add Our Models
        for model_name in MODEL_ORDER:
            if "SOTA" in model_name: continue

            search_terms = FILE_PATTERNS.get(model_name, [])
            matches = [f for f in all_files
                       if dataset_name in f.lower()
                       and any(t in f.lower() for t in search_terms)]

            for f in matches:
                full_path = os.path.join(target_dir, f)
                metrics = evaluate_embeddings(full_path)

                if metrics:
                    variant = parse_variant(f)
                    row = {
                        "Model": model_name,
                        "Variant": variant,
                        "Accuracy": metrics["Accuracy"],
                        "Top-3": metrics["Top-3"],
                        "F1 Macro": metrics["F1 Macro"],
                        "F1 Wtd": metrics["F1 Wtd"],
                        "Precision": metrics["Precision"],
                        "Recall": metrics["Recall"],
                        "Log Loss": metrics["Log Loss"]
                    }
                    table_rows.append(row)

        if not table_rows:
            print("   No results found for this dataset.")
            continue

        df = pd.DataFrame(table_rows)

        # Sort
        df['SortKey'] = df['Model'].apply(lambda x: 0 if "SOTA" in x else 1)
        df = df.sort_values(by=['SortKey', 'Model', 'Variant'])

        # Print Table
        print("-" * 110)
        print(f"{'Model':<15} {'Variant':<30} {'Acc':<7} {'Top3':<7} {'F1Mac':<7} {'LogLoss':<7}")
        print("-" * 110)
        for _, row in df.iterrows():
            acc = f"{float(row['Accuracy']):.4f}" if row['Accuracy'] != "-" else "-"
            top3 = f"{float(row['Top-3']):.4f}" if row['Top-3'] != "-" else "-"
            f1 = f"{float(row['F1 Macro']):.4f}" if row['F1 Macro'] != "-" else "-"
            ll = f"{float(row['Log Loss']):.4f}" if row['Log Loss'] != "-" else "-"
            print(f"{str(row['Model']):<15} {str(row['Variant']):<30} {acc:<7} {top3:<7} {f1:<7} {ll:<7}")
        print("-" * 110)

        # 3. GENERATE ALL 4 PLOTS
        print("   Generating plots...")
        try:
            plot_grouped_bar(df, dataset_name, target_dir)
            plot_radar_chart(df, dataset_name, target_dir)
            plot_heatmap(df, dataset_name, target_dir)
            plot_line_chart(df, dataset_name, target_dir)
        except Exception as e:
            print(f"   [Error] Plotting failed: {e}")
            import traceback
            traceback.print_exc()

        save_path = os.path.join(target_dir, f"final_results_{dataset_name}.csv")
        df.to_csv(save_path, index=False)
        print(f"   [Saved] CSV saved to: {save_path}")


if __name__ == "__main__":
    main()