import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
from tqdm import tqdm

# Parent directory for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_data(filepath):
    """Lädt die Embeddings aus den .pt Files."""
    try:
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def calculate_similarity_logits(train_vecs, train_labels, test_vecs, device="cpu"):
    """
    Berechnet die rohen Ähnlichkeits-Scores (Logits) basierend auf Zentroiden.
    Exakt wie in evaluation.py, aber OHNE Softmax.
    """
    unique_authors = sorted(list(set(train_labels)))
    centroids = []

    # 1. Zentroide berechnen
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

    centroid_matrix = torch.stack(centroids).to(device)
    test_vecs = test_vecs.to(device)

    # 2. Ähnlichkeit berechnen (Cosine Similarity)
    logits = torch.mm(test_vecs, centroid_matrix.transpose(0, 1))
    return logits, unique_authors


def find_optimal_t(logits, y_true_indices, n_classes):
    """
    Findet das T, das den LogLoss (NLL) minimiert.
    Nutzt Scipy für exakte Optimierung statt simpler Schleife.
    """
    logits_np = logits.numpy()

    # Zielfunktion: NLL in Abhängigkeit von T
    def nll_function(t):
        if t <= 0: return 999999  # Verhindert negative T

        # Scaling anwenden
        scaled_logits = logits_np / t

        # Softmax & LogLoss berechnen
        exp_vals = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        # Clip probabilities to avoid log(0)
        probs = np.clip(probs, 1e-15, 1 - 1e-15)

        loss = log_loss(y_true_indices, probs, labels=list(range(n_classes)))
        return loss

    # Suche T zwischen 0.1 und 50
    result = minimize_scalar(nll_function, bounds=(0.5, 50.0), method='bounded')
    return result.x, result.fun


def main():
    embeddings_dir = config.EMBEDDINGS_DIR
    print(f"--- TEMPERATURE CALIBRATION (Guo et al. 2017) ---")
    print(f"Scanning directory: {embeddings_dir}\n")

    files = [f for f in os.listdir(embeddings_dir) if f.endswith(".pt")]

    results = []

    for f in tqdm(files, desc="Calibrating"):
        path = os.path.join(embeddings_dir, f)
        data = load_data(path)
        if not data: continue

        # Daten extrahieren
        train_vecs = data.get('train_vecs')
        train_lbl = data.get('train_labels')
        test_vecs = data.get('test_vecs')
        test_lbl = data.get('test_labels')

        if train_vecs is None or test_vecs is None: continue

        # Labels in Indizes umwandeln
        unique_authors = sorted(list(set(train_lbl)))
        label_to_index = {name: i for i, name in enumerate(unique_authors)}
        y_true = [label_to_index[l] for l in test_lbl if l in label_to_index]

        # Logits berechnen
        logits, _ = calculate_similarity_logits(train_vecs, train_lbl, test_vecs)

        # Nur valide Testdaten nutzen
        valid_logits = logits[:len(y_true)]

        # OPTIMIERUNG
        best_t, best_loss = find_optimal_t(valid_logits, y_true, len(unique_authors))

        # Vergleichswert mit T=1.0 (Ohne Scaling)
        base_loss = log_loss(y_true, F.softmax(valid_logits, dim=1).numpy(), labels=list(range(len(unique_authors))))

        results.append({
            "File": f,
            "Optimal_T": best_t,
            "Base_Loss": base_loss,
            "Optimized_Loss": best_loss
        })

    # Ausgabe Tabelle
    print("\n" + "=" * 90)
    print(f"{'Model / File':<40} | {'Opt. T':<8} | {'Old Loss':<10} | {'New Loss':<10} | {'Improvement'}")
    print("-" * 90)

    avg_t_reuters = []
    avg_t_reddit = []

    for res in sorted(results, key=lambda x: x['File']):
        imp = res['Base_Loss'] - res['Optimized_Loss']
        fname = res['File'].replace(".pt", "")[:38]

        print(
            f"{fname:<40} | {res['Optimal_T']:.2f}     | {res['Base_Loss']:.4f}     | {res['Optimized_Loss']:.4f}     | -{imp:.4f}")

        if "reuters" in res['File']:
            avg_t_reuters.append(res['Optimal_T'])
        elif "darkreddit" in res['File']:
            avg_t_reddit.append(res['Optimal_T'])

    print("-" * 90)
    print(f"Average Optimal T (Reuters):    {np.mean(avg_t_reuters):.2f}")
    print(f"Average Optimal T (DarkReddit): {np.mean(avg_t_reddit):.2f}")
    print("=" * 90)


if __name__ == "__main__":
    main()