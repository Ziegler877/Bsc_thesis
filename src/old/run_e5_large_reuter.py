import os
import sys
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm  # Progress bar
from sklearn.metrics import accuracy_score, classification_report

# --- IMPORT CONFIG & SETUP PATHS ---
# Current location: src/old/
current_script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_script_dir)      # Points to 'src/' (where config.py is)
project_root = os.path.dirname(src_dir)            # Points to Project Root

sys.path.append(src_dir)       # Allows "import config"
sys.path.append(project_root)  # Allows access to data folders relative to root if needed

import config

# --- SETTINGS ---
# We use the official Hugging Face ID. It will download automatically (approx 2.5 GB).
# If you have it downloaded locally, you can replace this string with your local path.
MODEL_NAME = "intfloat/multilingual-e5-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on: {DEVICE}")


# --- HELPER FUNCTIONS ---
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def load_reuters_data(directory):
    texts = []
    labels = []

    if not os.path.exists(directory):
        print(f"(!) Directory not found: {directory}")
        return [], [], []

    authors = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

    print(f"Loading data from {os.path.basename(directory)}...")
    for author in tqdm(authors):
        author_path = os.path.join(directory, author)
        for filename in os.listdir(author_path):
            if filename.endswith(".txt"):
                try:
                    with open(os.path.join(author_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                        # E5 Requirement: Add "query: " prefix for classification tasks
                        content = "query: " + f.read()
                        texts.append(content)
                        labels.append(author)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
    return texts, labels, authors


def get_embeddings(model, tokenizer, texts, batch_size=16):
    """
    Generates embeddings.
    Note: Batch size is reduced to 16 (from 32) because E5-Large is bigger
    and uses more VRAM.
    """
    all_embeddings = []

    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding (E5-Large)"):
        batch_texts = texts[i: i + batch_size]

        # Tokenize
        batch_dict = tokenizer(
            batch_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**batch_dict)

        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # Normalize (Crucial for Cosine Similarity)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


# --- MAIN WORKFLOW ---

def main():
    # 1. Load Data
    print("--- 1. Loading Data ---")
    train_texts, train_labels, author_names = load_reuters_data(config.TRAIN_DIR)
    test_texts, test_labels, _ = load_reuters_data(config.TEST_DIR)

    if not train_texts:
        print("No data found. Check config.py paths.")
        return

    # 2. Load Model
    print(f"--- 2. Loading Model: {MODEL_NAME} ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    # 3. Generate Embeddings
    print("--- 3. Generating Training Embeddings ---")
    train_embeddings = get_embeddings(model, tokenizer, train_texts, batch_size=16)

    print("--- Generating Test Embeddings ---")
    test_embeddings = get_embeddings(model, tokenizer, test_texts, batch_size=16)

    # 4. Save Embeddings (New Filename)
    if not os.path.exists(config.PROCESSED_DIR):
        os.makedirs(config.PROCESSED_DIR)

    # IMPORTANT: Save as 'e5_large' so we don't overwrite 'e5_small' or 'llama'
    save_path = os.path.join(config.PROCESSED_DIR, "reuters_e5_large_data.pt")
    torch.save({
        'train_vecs': train_embeddings,
        'train_labels': train_labels,
        'test_vecs': test_embeddings,
        'test_labels': test_labels,
        'authors': author_names
    }, save_path)
    print(f"Saved processed data to {save_path}")

    # 5. Centroid Classification
    print("--- 4. Calculating Author Centroids ---")

    unique_authors = sorted(list(set(train_labels)))
    author_centroids = []
    label_to_index = {name: i for i, name in enumerate(unique_authors)}
    test_label_indices = torch.tensor([label_to_index[l] for l in test_labels])

    for author in unique_authors:
        indices = [i for i, x in enumerate(train_labels) if x == author]
        indices_tensor = torch.tensor(indices)

        author_vecs = train_embeddings.index_select(0, indices_tensor)

        # Calculate mean (centroid) and normalize
        centroid = torch.mean(author_vecs, dim=0)
        centroid = F.normalize(centroid, p=2, dim=0)
        author_centroids.append(centroid)

    # Stack into a single matrix (Shape: [50, 1024])
    centroid_matrix = torch.stack(author_centroids).to(DEVICE)

    # 6. Evaluation
    print("Evaluating...")
    test_embeddings = test_embeddings.to(DEVICE)

    # Calculate Similarity
    similarity_scores = torch.mm(test_embeddings, centroid_matrix.transpose(0, 1))
    predictions = torch.argmax(similarity_scores, dim=1).cpu()

    # 7. Metrics
    accuracy = accuracy_score(test_label_indices, predictions)
    print("================================================")
    print(f"Final Accuracy on Reuter 50-50 (E5-Large): {accuracy:.4f}")
    print("================================================")

    # Detailed report
    print(classification_report(test_label_indices, predictions, target_names=unique_authors))


if __name__ == "__main__":
    main()