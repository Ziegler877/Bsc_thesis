# src/old/data_loader_old.py
import os
from pathlib import Path


def load_reuters_data(base_path, split='C50train'):
    """
    [DEPRECATED] Old loader for Reuters 50-50 dataset.

    Reads the Reuters 50-50 dataset from a given base path.

    Args:
        base_path (str or Path): The path to the 'reuter+50+50' folder.
        split (str): 'C50train' or 'C50test'.

    Returns:
        texts (list): Content of articles
        labels (list): Author IDs (strings)
        authors (list): Unique author names sorted alphabetically
    """
    split_path = Path(base_path) / split
    texts = []
    labels = []

    # Check if path exists before trying to list directory
    if not split_path.exists():
        print(f"(!) Error: Path not found: {split_path}")
        return [], [], []

    # Get all author folders, sorted to ensure consistent label IDs
    author_folders = sorted([d for d in os.listdir(split_path) if (split_path / d).is_dir()])

    print(f"Loading {split} data from {split_path}...")

    for label_id, author in enumerate(author_folders):
        author_path = split_path / author
        for filename in os.listdir(author_path):
            file_path = author_path / filename
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    texts.append(content)
                    labels.append(author)  # Storing author name directly is often easier for analysis
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return texts, labels, author_folders