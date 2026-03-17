import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Parent directory for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_data(filepath):
    try:
        # Using weights_only=False to bypass PyTorch 2.6 block
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def parse_filename(filename):
    """Parses the .pt filename to extract readable configuration details for the table."""
    name = filename.replace(".pt", "")

    # 1. Extract Model
    if "e5_small" in name:
        model = "E5-Small"
    elif "e5_large" in name:
        model = "E5-Large"
    elif "llama2" in name:
        model = "Llama 2"
    elif "llama3" in name:
        model = "Llama 3"
    else:
        model = "Unknown"

    # 2. Extract Dataset
    if "reuters" in name:
        dataset = "Reuters"
    elif "darkreddit" in name:
        dataset = "DarkReddit"
    else:
        dataset = "Unknown"

    # 3. LoRA Status
    lora = "Yes" if "lora" in name else "No"

    # 4. Extract Extras (Chunking, Pooling, Run numbers, Sub-5)
    # Remove the known parts to leave only the unique config tags
    extras = name
    for term in ["e5_small", "e5_large", "llama2", "llama3", "reuters", "darkreddit", "lora"]:
        extras = extras.replace(term, "")

    # Clean up leftover underscores and format nicely
    extra_tags = [tag.upper() for tag in extras.split("_") if tag]
    config_details = " + ".join(extra_tags) if extra_tags else "Default/Base"

    return model, dataset, lora, config_details


def calculate_metrics(train_vecs, train_labels, test_vecs, test_labels):
    unique_authors = sorted(list(set(train_labels)))
    centroids = []

    # 1. Calculate Author Centroids
    for author in unique_authors:
        indices = [i for i, x in enumerate(train_labels) if x == author]
        if not indices:
            centroid = torch.zeros(train_vecs.shape[1])
        else:
            indices_tensor = torch.tensor(indices)
            author_vecs = train_vecs.index_select(0, indices_tensor)
            centroid = torch.mean(author_vecs, dim=0)
            centroid = torch.nan_to_num(centroid)
            centroid = F.normalize(centroid, p=2, dim=0)
        centroids.append(centroid)

    centroid_matrix = torch.stack(centroids)

    # 2. Calculate Distances (Cosine Similarity)
    logits = torch.mm(test_vecs, centroid_matrix.transpose(0, 1))

    # 3. Rank the predictions
    ranked_indices = torch.argsort(logits, dim=1, descending=True).numpy()

    label_to_index = {name: i for i, name in enumerate(unique_authors)}
    y_true_indices = [label_to_index[l] for l in test_labels if l in label_to_index]

    # 4. Calculate Top-5 and MRR
    top5_hits = 0
    reciprocal_ranks = []

    for i, true_idx in enumerate(y_true_indices):
        # Find what rank the true author got
        rank = np.where(ranked_indices[i] == true_idx)[0][0] + 1

        if rank <= 5:
            top5_hits += 1

        reciprocal_ranks.append(1.0 / rank)

    top5_accuracy = top5_hits / len(y_true_indices)
    mrr = np.mean(reciprocal_ranks)

    return top5_accuracy, mrr


def main():
    embeddings_dir = config.EMBEDDINGS_DIR
    print(f"=== CALCULATING ADVANCED METRICS (TOP-5 & MRR) ===")
    print(f"Scanning directory: {embeddings_dir}\n")

    files = [f for f in os.listdir(embeddings_dir) if f.endswith(".pt")]
    results = []

    for f in tqdm(files, desc="Processing Embeddings"):
        path = os.path.join(embeddings_dir, f)
        data = load_data(path)
        if not data: continue

        train_vecs = data.get('train_vecs')
        train_lbl = data.get('train_labels')
        test_vecs = data.get('test_vecs')
        test_lbl = data.get('test_labels')

        if train_vecs is None or test_vecs is None: continue

        # Parse filename for the table
        model, dataset, lora, config_str = parse_filename(f)

        # Calculate metrics
        top5, mrr = calculate_metrics(train_vecs, train_lbl, test_vecs, test_lbl)

        results.append({
            "File": f,
            "Model": model,
            "Dataset": dataset,
            "LoRA": lora,
            "Config": config_str,
            "Top5": top5 * 100,
            "MRR": mrr
        })

    # Sort results so the table is grouped logically (Model -> Dataset -> Config)
    results_sorted = sorted(results, key=lambda x: (x['Model'], x['Dataset'], x['Config']))

    # Print Huge Table
    print("\n" + "=" * 115)
    print(f"{'MODEL':<12} | {'DATASET':<12} | {'LORA':<5} | {'CONFIGURATION':<40} || {'TOP-5 ACC':<10} | {'MRR (MAP)'}")
    print("-" * 115)

    for res in results_sorted:
        print(
            f"{res['Model']:<12} | {res['Dataset']:<12} | {res['LoRA']:<5} | {res['Config']:<40} || {res['Top5']:>6.2f}%    | {res['MRR']:.4f}")

    print("=" * 115)


if __name__ == "__main__":
    main()