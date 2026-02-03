import os
import json
from tqdm import tqdm
import config


def load_dataset(dataset_name, split):
    """
    Args:
        dataset_name: 'reuters' or 'darkreddit'
        split: 'train' or 'test'
    """
    if dataset_name == "reuters":
        base_dir = config.REUTERS_TRAIN_DIR if split == "train" else config.REUTERS_TEST_DIR
        return _load_reuters_folders(base_dir)
    elif dataset_name == "darkreddit":
        filepath = config.DARK_REDDIT_TRAIN if split == "train" else config.DARK_REDDIT_TEST
        return _load_darkreddit_jsonl(filepath)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_reuters_folders(directory):
    texts, labels = [], []
    if not os.path.exists(directory):
        print(f"(!) Directory not found: {directory}")
        return [], []

    authors = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    print(f"   [Data] Loading Reuters ({os.path.basename(directory)})...")

    for author in tqdm(authors, leave=False):
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


def _load_darkreddit_jsonl(filepath):
    texts, labels = [], []
    if not os.path.exists(filepath):
        print(f"(!) File not found: {filepath}")
        return [], []

    print(f"   [Data] Loading DarkReddit ({os.path.basename(filepath)})...")

    # --- AUTO-DETECT KEYS ---
    # We read the first valid line to figure out the structure
    text_key = None
    author_key = None

    possible_text_keys = ['comment', 'body', 'text', 'content', 'selftext']
    possible_author_keys = ['author', 'author_id', 'user_id', 'username']

    with open(filepath, 'r', encoding='utf-8') as f:
        # Find first valid JSON line to inspect
        for line in f:
            try:
                sample = json.loads(line)
                # Check which keys exist in this sample
                keys = sample.keys()

                # Find the matching text key
                for k in possible_text_keys:
                    if k in keys:
                        text_key = k
                        break

                # Find the matching author key
                for k in possible_author_keys:
                    if k in keys:
                        author_key = k
                        break

                if text_key and author_key:
                    print(f"   [Info] Detected keys - Text: '{text_key}', Author: '{author_key}'")
                    break
            except:
                continue

        # Reset file pointer to beginning to read data
        f.seek(0)

        if not text_key or not author_key:
            print(f"(!) Could not auto-detect keys. Checked: {possible_text_keys}")
            return [], []

        # Load Data
        for line in tqdm(f, leave=False):
            try:
                data = json.loads(line)
                content = data.get(text_key, "").strip()
                author = data.get(author_key, "unknown")

                # Filter out deleted/empty
                if content and author not in ["[deleted]", "unknown"]:
                    texts.append(content)
                    labels.append(author)
            except:
                continue

    return texts, labels