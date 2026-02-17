import os
import json
import random
from collections import defaultdict
from tqdm import tqdm
import config


def load_dataset(dataset_name, split, subset_size=None):
    """
    Args:
        dataset_name: 'reuters' or 'darkreddit'
        split: 'train' or 'test'
        subset_size: (Optional) Int. If set, limits texts per author to this number (randomly selected).
    """
    if dataset_name == "reuters":
        base_dir = config.REUTERS_TRAIN_DIR if split == "train" else config.REUTERS_TEST_DIR
        return _load_reuters_folders(base_dir, subset_size=subset_size)
    elif dataset_name == "darkreddit":
        filepath = config.DARK_REDDIT_TRAIN if split == "train" else config.DARK_REDDIT_TEST
        return _load_darkreddit_jsonl(filepath, subset_size=subset_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_reuters_folders(directory, subset_size=None):
    texts, labels = [], []
    if not os.path.exists(directory):
        print(f"(!) Directory not found: {directory}")
        return [], []

    authors = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

    info_str = f" (Subset: {subset_size}/author)" if subset_size else ""
    print(f"   [Data] Loading Reuters ({os.path.basename(directory)}){info_str}...")

    for author in tqdm(authors, leave=False):
        author_path = os.path.join(directory, author)

        # Get all valid text files
        filenames = [f for f in os.listdir(author_path) if f.endswith(".txt")]

        # --- SUBSET LOGIC ---
        if subset_size is not None:
            random.shuffle(filenames)  # Shuffle to ensure randomness
            filenames = filenames[:subset_size]  # Take only N
        # --------------------

        for filename in filenames:
            try:
                with open(os.path.join(author_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())
                    labels.append(author)
            except:
                pass
    return texts, labels


def _load_darkreddit_jsonl(filepath, subset_size=None):
    texts, labels = [], []
    if not os.path.exists(filepath):
        print(f"(!) File not found: {filepath}")
        return [], []

    info_str = f" (Subset: {subset_size}/author)" if subset_size else ""
    print(f"   [Data] Loading DarkReddit ({os.path.basename(filepath)}){info_str}...")

    # --- AUTO-DETECT KEYS ---
    text_key = None
    author_key = None
    possible_text_keys = ['comment', 'body', 'text', 'content', 'selftext']
    possible_author_keys = ['author', 'author_id', 'user_id', 'username']

    with open(filepath, 'r', encoding='utf-8') as f:
        # Inspect first line for keys
        for line in f:
            try:
                sample = json.loads(line)
                keys = sample.keys()
                for k in possible_text_keys:
                    if k in keys:
                        text_key = k
                        break
                for k in possible_author_keys:
                    if k in keys:
                        author_key = k
                        break
                if text_key and author_key:
                    print(f"   [Info] Detected keys - Text: '{text_key}', Author: '{author_key}'")
                    break
            except:
                continue

        f.seek(0)
        if not text_key or not author_key:
            print(f"(!) Could not auto-detect keys. Checked: {possible_text_keys}")
            return [], []

        # Load ALL Data first (necessary for random sampling from stream)
        for line in tqdm(f, leave=False):
            try:
                data = json.loads(line)
                content = data.get(text_key, "").strip()
                author = data.get(author_key, "unknown")

                if content and author not in ["[deleted]", "unknown"]:
                    texts.append(content)
                    labels.append(author)
            except:
                continue

    # --- SUBSET LOGIC (Post-Processing) ---
    if subset_size is not None:
        print(f"   [Data] Subsampling to {subset_size} random texts per author...")

        # 1. Group by Author
        author_map = defaultdict(list)
        for t, l in zip(texts, labels):
            author_map[l].append(t)

        # 2. Sample and Flatten
        texts, labels = [], []
        for author, user_texts in author_map.items():
            if len(user_texts) > subset_size:
                selected = random.sample(user_texts, subset_size)
            else:
                selected = user_texts

            texts.extend(selected)
            labels.extend([author] * len(selected))

        print(f"   [Data] Final size after subsampling: {len(texts)}")
    # --------------------------------------

    return texts, labels