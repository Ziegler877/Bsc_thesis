import os
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm  # Progress bar
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION & PATH SETUP ---
# Current location: src/old/
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to 'src/'
src_dir = os.path.dirname(current_script_dir)

# Go up another level to the Project Root
project_root = os.path.dirname(src_dir)

# Define Data Paths
RAW_DATA_PATH = os.path.join(project_root, "data", "raw", "reuter+50+50")
TRAIN_DIR = os.path.join(RAW_DATA_PATH, "C50train")
TEST_DIR = os.path.join(RAW_DATA_PATH, "C50test")
PROCESSED_DIR = os.path.join(project_root, "data", "processed")

# Define Model Path (Loading from your local "models" folder)
MODEL_PATH = os.path.join(project_root, "models", "e5-small")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on: {DEVICE}")
print(f"Loading Model from: {MODEL_PATH}")


# --- HELPER FUNCTIONS ---
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def load_reuters_data(directory):
    """
    Reads all text files from the directory.
    Structure: directory/AuthorName/file.txt
    """
    texts = []
    labels = []

    # Get author folders
    if not os.path.exists(directory):
        print(f"(!) Directory not found: {directory}")
        return [], [], []

    authors = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

    print(f"Loading data from {os.path.basename(directory)}...")
    for author in tqdm(authors):
        author_path = os.path.join(directory, author)
        for filename in os.listdir(author_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(author_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        # E5 Requirement: Add "query: " prefix for classification tasks
                        content = "query: " + f.read()
                        texts.append(content)
                        labels.append(author)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
    return texts, labels, authors


def get_embeddings(model, tokenizer, texts, batch_size=32):
    """
    Generates embeddings for a list of texts using batch processing.
    """
    all_embeddings = []

    model.eval()  # Set model to evaluation mode

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
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
    train_texts, train_labels, author_names = load_reuters_data(TRAIN_DIR)
    test_texts, test_labels, _ = load_reuters_data(TEST_DIR)

    if not train_texts:
        print("No training data found. Check paths.")
        return

    # 2. Load Local Model
    print(f"--- 2. Loading Local Model ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
    except OSError:
        print(f"Could not load local model from {MODEL_PATH}.")
        print(
            "Please ensure the folder structure is 'models/e5-small' and contains 'config.json' and 'model.safetensors' (or .bin)")
        return

    # 3. Generate Embeddings (The "Feature Extraction")
    print("--- 3. Generating Training Embeddings ---")
    train_embeddings = get_embeddings(model, tokenizer, train_texts)

    print("--- Generating Test Embeddings ---")
    test_embeddings = get_embeddings(model, tokenizer, test_texts)

    # --- SAVE EMBEDDINGS (NEW) ---
    # We save these so you can use them later without re-running the slow model
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    save_path = os.path.join(PROCESSED_DIR, "reuters_e5_small_data.pt")
    torch.save({
        'train_vecs': train_embeddings,
        'train_labels': train_labels,
        'test_vecs': test_embeddings,
        'test_labels': test_labels,
        'authors': author_names
    }, save_path)
    print(f"Saved processed data to {save_path}")

    # 4. "Training" Phase: Calculate Author Centroids (Prototypes)
    print("--- 4. Calculating Author Centroids ---")

    unique_authors = sorted(list(set(train_labels)))
    author_centroids = []

    # Map author names to numeric indices
    label_to_index = {name: i for i, name in enumerate(unique_authors)}
    train_label_indices = torch.tensor([label_to_index[l] for l in train_labels])
    test_label_indices = torch.tensor([label_to_index[l] for l in test_labels])

    for author in unique_authors:
        # Find all indices for this author in the training set
        indices = [i for i, x in enumerate(train_labels) if x == author]
        indices_tensor = torch.tensor(indices)

        # Get all embeddings for this author
        author_vecs = train_embeddings.index_select(0, indices_tensor)

        # Calculate mean (centroid) and normalize again
        centroid = torch.mean(author_vecs, dim=0)
        centroid = F.normalize(centroid, p=2, dim=0)
        author_centroids.append(centroid)

    # Stack into a single matrix (Shape: [50, 384])
    centroid_matrix = torch.stack(author_centroids).to(DEVICE)

    # 5. Evaluation Phase
    print("Evaluating...")
    test_embeddings = test_embeddings.to(DEVICE)

    # Calculate Similarity: Test_Vecs @ Centroids_Transposed
    similarity_scores = torch.mm(test_embeddings, centroid_matrix.transpose(0, 1))

    # The predicted author is the one with the highest similarity score
    predictions = torch.argmax(similarity_scores, dim=1).cpu()

    # 6. Metrics
    accuracy = accuracy_score(test_label_indices, predictions)
    print("================================================")
    print(f"Final Accuracy on Reuter 50-50 (E5-Small): {accuracy:.4f}")
    print("================================================")

    # Print a condensed report
    print(classification_report(test_label_indices, predictions, target_names=unique_authors))


if __name__ == "__main__":
    main()