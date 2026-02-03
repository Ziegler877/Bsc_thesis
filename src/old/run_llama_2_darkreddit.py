import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# --- IMPORT CONFIG & SETUP PATHS ---
# Current location: src/old/
current_script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_script_dir)      # Points to 'src/' (where config.py is)
project_root = os.path.dirname(src_dir)            # Points to Project Root

sys.path.append(src_dir)       # Allows "import config"
sys.path.append(project_root)  # Allows access to data folders relative to root if needed

import config

# --- SETTINGS ---
# Use the Auto-Download path from config (NousResearch/Llama-2-7b-hf)
MODEL_PATH = config.LLAMA_MODEL_PATH
DEVICE = "cuda"

print(f"Running Llama-2 on: {DEVICE}")


# --- HELPER FUNCTIONS ---
def detect_jsonl_keys(filepath):
    """
    Reads the first line to auto-detect json keys.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            first_line = json.loads(f.readline())
        except:
            return None, None

    keys = first_line.keys()
    text_key = next((k for k in ['body', 'text', 'content', 'comment'] if k in keys), None)
    author_key = next((k for k in ['author', 'author_id', 'label', 'username'] if k in keys), None)

    print(f"   [Detected Keys] Text: '{text_key}' | Author: '{author_key}'")
    return text_key, author_key


def load_darkreddit_jsonl(filepath, limit=None):
    texts = []
    labels = []

    if not os.path.exists(filepath):
        print(f"(!) File not found: {filepath}")
        return [], [], []

    print(f"Loading data from {os.path.basename(filepath)}...")
    text_key, author_key = detect_jsonl_keys(filepath)
    if not text_key: return [], [], []

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line)
                content = data.get(text_key, "").strip()
                author = data.get(author_key, "unknown")

                # Filter bad data
                if not content or author in ["[deleted]", "unknown"]:
                    continue

                # Note: Llama usually doesn't need "query: " prefix like E5,
                # but adding it doesn't hurt. We'll stick to raw text for Llama.
                texts.append(content)
                labels.append(author)

            except json.JSONDecodeError:
                continue

    unique_authors = sorted(list(set(labels)))
    print(f"   Loaded {len(texts)} texts from {len(unique_authors)} unique authors.")
    return texts, labels, unique_authors


def get_llama_embeddings(model, tokenizer, texts, batch_size=2):
    """
    Extracts embeddings using the LAST token hidden state.
    Batch size = 2 (Tight fit for 6GB VRAM).
    """
    all_embeddings = []
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(texts), batch_size), desc="Llama Embedding"):
        batch_texts = texts[i: i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get last hidden state of the last token
        last_hidden_state = outputs.hidden_states[-1]
        embeddings = last_hidden_state[:, -1, :]

        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


# --- MAIN ---
def main():
    # 1. Load Data
    print("--- 1. Loading DarkReddit Data ---")
    train_texts, train_labels, _ = load_darkreddit_jsonl(config.DARK_REDDIT_TRAIN)
    test_texts, test_labels, unique_authors = load_darkreddit_jsonl(config.DARK_REDDIT_TEST)

    if not train_texts:
        return

    # Filter Test set (Remove authors not in training data)
    train_authors_set = set(train_labels)
    filtered_test_texts = []
    filtered_test_labels = []

    for t, l in zip(test_texts, test_labels):
        if l in train_authors_set:
            filtered_test_texts.append(t)
            filtered_test_labels.append(l)

    test_texts = filtered_test_texts
    test_labels = filtered_test_labels
    print(f"   Final Test Size (Filtered): {len(test_texts)}")

    # 2. Load Llama-2 (4-bit)
    print(f"--- 2. Loading Llama-2 from: {MODEL_PATH} ---")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    try:
        # Load Tokenizer (Download if needed)
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        except:
            print("(!) Using fallback tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

        # Load Model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Generate Embeddings
    # Warning: Llama is slow.
    print("--- 3. Generating Embeddings ---")
    train_embeddings = get_llama_embeddings(model, tokenizer, train_texts, batch_size=2)
    test_embeddings = get_llama_embeddings(model, tokenizer, test_texts, batch_size=2)

    # 4. Save
    if not os.path.exists(config.PROCESSED_DIR):
        os.makedirs(config.PROCESSED_DIR)

    save_path = os.path.join(config.PROCESSED_DIR, "darkreddit_llama2_data.pt")
    torch.save({
        'train_vecs': train_embeddings,
        'train_labels': train_labels,
        'test_vecs': test_embeddings,
        'test_labels': test_labels
    }, save_path)
    print(f"Saved processed data to {save_path}")

    # 5. Centroid Classification
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

    centroid_matrix = torch.stack(author_centroids).to("cpu")

    # Eval
    # Move to CPU for similarity calc to save VRAM
    scores = torch.mm(test_embeddings, centroid_matrix.transpose(0, 1))
    preds = torch.argmax(scores, dim=1).cpu()

    true_indices = [label_to_index[l] for l in test_labels]

    acc = accuracy_score(true_indices, preds)
    print("================================================")
    print(f"Final Accuracy on DarkReddit (Llama-2): {acc:.4f}")
    print("================================================")


if __name__ == "__main__":
    main()