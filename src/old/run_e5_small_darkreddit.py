import os
import sys
import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
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
MODEL_NAME = "intfloat/multilingual-e5-small"  # Using Small for speed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on: {DEVICE}")


# --- HELPER FUNCTIONS ---
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def detect_jsonl_keys(filepath):
    """
    Reads the first line to figure out what the keys are (e.g. 'body' vs 'text').
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = json.loads(f.readline())

    keys = first_line.keys()

    # 1. Find Text Key
    text_key = None
    for k in ['body', 'text', 'content', 'comment', 'selftext']:
        if k in keys:
            text_key = k
            break

    # 2. Find Author Key
    author_key = None
    for k in ['author', 'author_id', 'label', 'username', 'user_id']:
        if k in keys:
            author_key = k
            break

    if not text_key or not author_key:
        raise ValueError(f"Could not auto-detect keys in {filepath}. Found: {list(keys)}")

    print(f"   [Detected Keys] Text: '{text_key}' | Author: '{author_key}'")
    return text_key, author_key


def load_darkreddit_jsonl(filepath, limit=None):
    texts = []
    labels = []

    if not os.path.exists(filepath):
        print(f"(!) File not found: {filepath}")
        return [], [], []

    print(f"Loading data from {os.path.basename(filepath)}...")

    # Auto-detect keys
    text_key, author_key = detect_jsonl_keys(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line)

                # Extract content
                content = data.get(text_key, "").strip()
                author = data.get(author_key, "unknown")

                # Skip empty texts or deleted authors
                if not content or author in ["[deleted]", "unknown"]:
                    continue

                # E5 Requirement: Add "query: "
                texts.append("query: " + content)
                labels.append(author)

            except json.JSONDecodeError:
                continue

    unique_authors = sorted(list(set(labels)))
    print(f"   Loaded {len(texts)} texts from {len(unique_authors)} unique authors.")
    return texts, labels, unique_authors


def get_embeddings(model, tokenizer, texts, batch_size=32):
    all_embeddings = []
    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i: i + batch_size]

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
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


# --- MAIN ---
def main():
    # 1. Load Data
    print("--- 1. Loading DarkReddit Data ---")

    # Optional: Set limit=1000 if you just want to test if it works first
    train_texts, train_labels, _ = load_darkreddit_jsonl(config.DARK_REDDIT_TRAIN)
    test_texts, test_labels, unique_authors = load_darkreddit_jsonl(config.DARK_REDDIT_TEST)

    if not train_texts:
        print("No training data found. Check paths.")
        return

    # Filter Test Set: Only keep authors that exist in the Training Set
    # (Attribution only works if we have seen the author before!)
    train_authors_set = set(train_labels)

    print("Filtering Test set (removing unknown authors)...")
    filtered_test_texts = []
    filtered_test_labels = []

    for t, l in zip(test_texts, test_labels):
        if l in train_authors_set:
            filtered_test_texts.append(t)
            filtered_test_labels.append(l)

    test_texts = filtered_test_texts
    test_labels = filtered_test_labels
    print(f"   Final Test Size: {len(test_texts)} texts")

    # 2. Load Model
    print(f"--- 2. Loading Model: {MODEL_NAME} ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    # 3. Embed
    print("--- 3. Generating Embeddings ---")
    train_embeddings = get_embeddings(model, tokenizer, train_texts)
    test_embeddings = get_embeddings(model, tokenizer, test_texts)

    # 4. Save
    if not os.path.exists(config.PROCESSED_DIR):
        os.makedirs(config.PROCESSED_DIR)

    torch.save({
        'train_vecs': train_embeddings,
        'train_labels': train_labels,
        'test_vecs': test_embeddings,
        'test_labels': test_labels
    }, os.path.join(config.PROCESSED_DIR, "darkreddit_e5_small.pt"))

    # 5. Centroids & Score
    print("--- 4. Classification ---")
    unique_authors = sorted(list(set(train_labels)))
    label_to_index = {name: i for i, name in enumerate(unique_authors)}

    author_centroids = []
    for author in unique_authors:
        indices = [i for i, x in enumerate(train_labels) if x == author]
        if not indices: continue
        indices_tensor = torch.tensor(indices)
        author_vecs = train_embeddings.index_select(0, indices_tensor)
        centroid = F.normalize(torch.mean(author_vecs, dim=0), p=2, dim=0)
        author_centroids.append(centroid)

    centroid_matrix = torch.stack(author_centroids).to(DEVICE)

    # Eval
    test_embeddings = test_embeddings.to(DEVICE)
    scores = torch.mm(test_embeddings, centroid_matrix.transpose(0, 1))
    preds = torch.argmax(scores, dim=1).cpu()

    # Map string labels to indices
    true_indices = [label_to_index[l] for l in test_labels]

    acc = accuracy_score(true_indices, preds)
    print("================================================")
    print(f"Final Accuracy on DarkReddit (E5-Small): {acc:.4f}")
    print("================================================")


if __name__ == "__main__":
    main()