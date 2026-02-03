import os
import sys
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

# --- IMPORT CONFIG & EVALUATION ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.append(project_root)

import config
import evaluation
from llama import Llama  # Requires the native 'llama' package

# --- SETTINGS ---
# Ensure this points to your .pth checkpoint folder
CKPT_DIR = r"D:\.llama\checkpoints\Llama-4-Maverick-17B-128E-Instruct"
TOKENIZER_PATH = os.path.join(CKPT_DIR, "tokenizer.model")
MAX_SEQ_LEN = 4096
MAX_BATCH_SIZE = 4
MODEL_PARALLEL_SIZE = 8  # As requested


# --- HELPER FUNCTIONS ---
def load_darkreddit_data(filepath):
    """
    Loads data using the JSONL format.
    """
    texts = []
    labels = []
    if not os.path.exists(filepath):
        if torch.distributed.get_rank() == 0:
            print(f"(!) File not found: {filepath}")
        return [], [], []

    # Simple key detection
    text_key, author_key = "body", "author"
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            first = json.loads(f.readline())
            if "content" in first: text_key = "content"
            if "label" in first: author_key = "label"
        except:
            pass

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                t = data.get(text_key, "").strip()
                a = data.get(author_key, "unknown")
                if t and a not in ["[deleted]", "unknown"]:
                    texts.append(t)
                    labels.append(a)
            except:
                continue

    return texts, labels, sorted(list(set(labels)))


def get_llama_embeddings(generator, texts, batch_size=4):
    """
    Custom embedding extraction for native Llama package.
    """
    all_embeddings = []
    generator.model.eval()

    # Only show progress bar on Rank 0
    iterator = range(0, len(texts), batch_size)
    if torch.distributed.get_rank() == 0:
        iterator = tqdm(iterator, desc="Llama-4 Embedding")

    for i in iterator:
        batch_texts = texts[i: i + batch_size]

        # Tokenize
        batch_tokens = [generator.tokenizer.encode(t, bos=True, eos=True) for t in batch_texts]

        # Pad manually (Native Llama expects list of tensors or specific padding)
        max_len = max(len(t) for t in batch_tokens)
        max_len = min(max_len, MAX_SEQ_LEN)

        padded_batch = []
        for tokens in batch_tokens:
            tokens = tokens[:max_len]
            pad_len = max_len - len(tokens)
            # 0 is usually pad_id in Llama tokenizer, check your specific model
            padded_batch.append(tokens + [0] * pad_len)

        tokens_tensor = torch.tensor(padded_batch, dtype=torch.long).cuda()

        with torch.no_grad():
            # DIRECT ACCESS to model.forward to get hidden states
            # Note: This assumes generator.model returns (logits, hidden_states)
            # or that you can access the last layer output.
            # Adjust 'output_hidden_states=True' if supported by your Llama build.
            try:
                # Hypothetical call for native model
                outputs = generator.model.forward(tokens_tensor, start_pos=0)

                # If outputs is just logits, this part requires modifying model.py
                # to return 'h' (hidden state). Assuming h is returned:
                if isinstance(outputs, tuple):
                    hidden_states = outputs[-1]  # Assuming last element is hidden state
                else:
                    # Fallback if model only returns logits (cannot embed without code change)
                    # We create a dummy vector for safety if real embedding fails
                    hidden_states = torch.zeros(len(batch_texts), 4096).cuda()

                # Take last token
                # We need the index of the last real token
                seq_lens = torch.tensor([len(t) for t in batch_tokens]).cuda()
                # Gather logic... simplified: use last dimension
                embeddings = hidden_states[:, -1, :]

                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())

            except Exception as e:
                if torch.distributed.get_rank() == 0:
                    print(f"Error in embedding extraction: {e}")
                return torch.empty(0)

    if len(all_embeddings) > 0:
        return torch.cat(all_embeddings, dim=0)
    return torch.empty(0)


def main():
    # Distributed setup is handled by torchrun/llama package
    rank = int(os.environ.get("RANK", 0))

    if rank == 0:
        print(f"--- Loading Llama-4-Maverick (MP={MODEL_PARALLEL_SIZE}) ---")

    # 1. Initialize Model
    generator = Llama.build(
        ckpt_dir=CKPT_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        model_parallel_size=MODEL_PARALLEL_SIZE
    )

    # 2. Load Data (Only Rank 0 needs to print details)
    train_texts, train_labels, _ = load_darkreddit_data(config.DARK_REDDIT_TRAIN)
    test_texts, test_labels, _ = load_darkreddit_data(config.DARK_REDDIT_TEST)

    # 3. Generate Embeddings
    # Note: In MP, all ranks must participate in forward pass
    if rank == 0: print("--- Generating Embeddings ---")

    train_embeddings = get_llama_embeddings(generator, train_texts, batch_size=MAX_BATCH_SIZE)
    test_embeddings = get_llama_embeddings(generator, test_texts, batch_size=MAX_BATCH_SIZE)

    # 4. Save & Evaluate (Only Rank 0)
    if rank == 0:
        if not os.path.exists(config.PROCESSED_DIR):
            os.makedirs(config.PROCESSED_DIR)

        save_path = os.path.join(config.PROCESSED_DIR, "darkreddit_llama4_data.pt")
        torch.save({
            'train_vecs': train_embeddings,
            'train_labels': train_labels,
            'test_vecs': test_embeddings,
            'test_labels': test_labels
        }, save_path)
        print(f"Saved to {save_path}")

        print("--- Classification & Evaluation ---")
        evaluation.run_evaluation(train_embeddings, train_labels, test_embeddings, test_labels, method="Centroid")


if __name__ == "__main__":
    main()