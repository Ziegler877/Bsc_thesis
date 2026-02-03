import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from llama import Llama

# --- IMPORT CONFIG & EVALUATION ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.append(project_root)

import config
import evaluation

# --- SETTINGS ---
CKPT_DIR = r"D:\.llama\checkpoints\Llama-4-Maverick-17B-128E-Instruct"
TOKENIZER_PATH = os.path.join(CKPT_DIR, "tokenizer.model")
MAX_SEQ_LEN = 4096
MAX_BATCH_SIZE = 4
MODEL_PARALLEL_SIZE = 8


def load_reuters_data(directory):
    texts = []
    labels = []
    if not os.path.exists(directory): return [], [], []

    authors = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

    # Progress bar only on rank 0 usually, simplified here
    for author in authors:
        author_path = os.path.join(directory, author)
        for filename in os.listdir(author_path):
            if filename.endswith(".txt"):
                try:
                    with open(os.path.join(author_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                        texts.append(f.read())
                        labels.append(author)
                except:
                    pass
    return texts, labels


def get_llama_embeddings(generator, texts, batch_size=4):
    all_embeddings = []
    generator.model.eval()

    iterator = range(0, len(texts), batch_size)
    if torch.distributed.get_rank() == 0:
        iterator = tqdm(iterator, desc="Llama-4 Embedding")

    for i in iterator:
        batch_texts = texts[i: i + batch_size]
        batch_tokens = [generator.tokenizer.encode(t, bos=True, eos=True) for t in batch_texts]

        max_len = max(len(t) for t in batch_tokens)
        max_len = min(max_len, MAX_SEQ_LEN)

        padded_batch = []
        for tokens in batch_tokens:
            tokens = tokens[:max_len]
            padded_batch.append(tokens + [0] * (max_len - len(tokens)))

        tokens_tensor = torch.tensor(padded_batch, dtype=torch.long).cuda()

        with torch.no_grad():
            try:
                # Assuming model returns (logits, hidden_states) or last layer output
                outputs = generator.model.forward(tokens_tensor, start_pos=0)

                # ADAPT THIS LINE based on your specific Llama-4 model.py implementation
                # If outputs is a tuple, index [-1] is usually hidden states.
                hidden_states = outputs[-1] if isinstance(outputs, tuple) else outputs

                embeddings = hidden_states[:, -1, :]
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())
            except:
                # Return zeros if model structure doesn't support embedding extraction
                all_embeddings.append(torch.zeros(len(batch_texts), 4096))

    return torch.cat(all_embeddings, dim=0)


def main():
    rank = int(os.environ.get("RANK", 0))

    if rank == 0:
        print(f"--- Loading Llama-4-Maverick (MP={MODEL_PARALLEL_SIZE}) ---")

    generator = Llama.build(
        ckpt_dir=CKPT_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        model_parallel_size=MODEL_PARALLEL_SIZE
    )

    # Load Data
    train_texts, train_labels = load_reuters_data(config.TRAIN_DIR)
    test_texts, test_labels = load_reuters_data(config.TEST_DIR)

    if rank == 0: print(f"Loaded {len(train_texts)} training docs.")

    # Embed
    train_embeddings = get_llama_embeddings(generator, train_texts, batch_size=MAX_BATCH_SIZE)
    test_embeddings = get_llama_embeddings(generator, test_texts, batch_size=MAX_BATCH_SIZE)

    # Save & Eval
    if rank == 0:
        if not os.path.exists(config.PROCESSED_DIR):
            os.makedirs(config.PROCESSED_DIR)

        save_path = os.path.join(config.PROCESSED_DIR, "reuters_llama4_data.pt")
        torch.save({
            'train_vecs': train_embeddings,
            'train_labels': train_labels,
            'test_vecs': test_embeddings,
            'test_labels': test_labels
        }, save_path)

        print("--- Classification & Evaluation ---")
        evaluation.run_evaluation(train_embeddings, train_labels, test_embeddings, test_labels, method="Centroid")


if __name__ == "__main__":
    main()