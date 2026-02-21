import os
import sys
import json
from tqdm import tqdm

# Fix path to access root config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

import config


def load_train_val_data(dataset_name):
    """
    Loads training and validation data strictly from defined files/folders.
    Returns: train_texts, val_texts, train_labels, val_labels
    """
    print(f"   [Data] Loading {dataset_name} (Strict File Mode)...")

    if dataset_name == "reuters":
        val_dir = getattr(config, 'REUTERS_VAL_DIR', None)
        if not val_dir or not os.path.exists(val_dir):
            raise FileNotFoundError(
                f"Reuters validation folder not found at: {val_dir}. Please run split_data.py first.")

        print("   [Data] Loading Reuters Train and Validation folders...")
        train_texts, train_labels = _load_reuters_folder(config.REUTERS_TRAIN_DIR)
        val_texts, val_labels = _load_reuters_folder(val_dir)
        return train_texts, val_texts, train_labels, val_labels

    elif dataset_name == "darkreddit":
        val_file = getattr(config, 'DARK_REDDIT_VAL', None)
        if not val_file or not os.path.exists(val_file):
            raise FileNotFoundError(f"DarkReddit validation file not found at: {val_file}.")

        print("   [Data] Loading DarkReddit Train and Validation files...")
        train_texts, train_labels = _load_darkreddit_jsonl(config.DARK_REDDIT_TRAIN)
        val_texts, val_labels = _load_darkreddit_jsonl(val_file)
        return train_texts, val_texts, train_labels, val_labels

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_reuters_folder(directory):
    texts, labels = [], []
    if not os.path.exists(directory):
        print(f"(!) Directory not found: {directory}")
        return [], []

    authors = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

    for author in tqdm(authors, desc=f"Reading Reuters ({os.path.basename(directory)})"):
        author_path = os.path.join(directory, author)
        filenames = [f for f in os.listdir(author_path) if f.endswith(".txt")]
        for filename in filenames:
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

    text_key, author_key = None, None
    possible_text_keys = ['comment', 'body', 'text', 'content', 'selftext']
    possible_author_keys = ['author', 'author_id', 'user_id', 'username']

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line)
                keys = sample.keys()
                for k in possible_text_keys:
                    if k in keys: text_key = k
                for k in possible_author_keys:
                    if k in keys: author_key = k
                if text_key and author_key: break
            except:
                continue

        f.seek(0)
        if not text_key or not author_key: return [], []

        for line in tqdm(f, desc=f"Reading JSONL ({os.path.basename(filepath)})"):
            try:
                data = json.loads(line)
                content = data.get(text_key, "").strip()
                author = data.get(author_key, "unknown")
                if content and author not in ["[deleted]", "unknown"]:
                    texts.append(content)
                    labels.append(author)
            except:
                continue

    return texts, labels