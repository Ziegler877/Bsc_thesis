import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

# --- IMPORT CONFIG & EVALUATION ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir) # Points to src/
sys.path.append(project_root)

import config
import evaluation  # Importing the shared evaluation module


# --- HELPER FUNCTIONS ---
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
                        texts.append(f.read())
                        labels.append(author)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
    return texts, labels, authors


def get_llama_embeddings(model, tokenizer, texts, batch_size=2):
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
        ).to(config.DEVICE)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get last hidden state
        last_hidden_state = outputs.hidden_states[-1]

        # Take the vector of the last token
        embeddings = last_hidden_state[:, -1, :]

        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


# --- MAIN ---
def main():
    # 1. Load Data using Config Paths
    print("--- 1. Loading Data ---")
    train_texts, train_labels, author_names = load_reuters_data(config.TRAIN_DIR)
    test_texts, test_labels, _ = load_reuters_data(config.TEST_DIR)

    if not train_texts:
        print("No training data found. Check paths in config.py")
        return

    # 2. Load Llama-2 with 4-bit Quantization
    print(f"--- 2. Loading Llama-2 from: {config.LLAMA_MODEL_PATH} ---")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    try:
        # Try loading tokenizer from local folder first
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH)
        except:
            print("(!) Could not find tokenizer in local folder. Downloading default Llama-2 tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

        # Load Model
        model = AutoModelForCausalLM.from_pretrained(
            config.LLAMA_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto"
        )
    except Exception as e:
        print(f"\nCRITICAL ERROR Loading Model:\n{e}")
        print("\nPossible fixes:")
        print("1. Ensure your D: drive path contains 'config.json' and 'model.safetensors' (HuggingFace format).")
        print("2. If you have raw Meta weights (.pth files), you must convert them to HF format first.")
        return

    # 3. Generate Embeddings
    print("--- 3. Generating Embeddings ---")
    train_embeddings = get_llama_embeddings(model, tokenizer, train_texts, batch_size=2)
    test_embeddings = get_llama_embeddings(model, tokenizer, test_texts, batch_size=2)

    # 4. Save Results
    if not os.path.exists(config.PROCESSED_DIR):
        os.makedirs(config.PROCESSED_DIR)

    save_path = os.path.join(config.PROCESSED_DIR, "reuters_llama2_data.pt")
    torch.save({
        'train_vecs': train_embeddings,
        'train_labels': train_labels,
        'test_vecs': test_embeddings,
        'test_labels': test_labels
    }, save_path)
    print(f"Saved embeddings to {save_path}")

    # 5. Evaluation (Using Shared Module)
    print("--- 4. Classification & Evaluation ---")
    evaluation.run_evaluation(train_embeddings, train_labels, test_embeddings, test_labels, method="Centroid")


if __name__ == "__main__":
    main()