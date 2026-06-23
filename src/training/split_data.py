import os
import shutil
import random
import hashlib
import sys

# Connect to config
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(src_dir)
sys.path.insert(0, root_dir)

import config


def get_file_hash(filepath):
    """Calculates the MD5 hash of a file to perfectly identify duplicate content."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception as e:
        return None


def analyze_overlap(train_dir, test_dir):
    """Checks if any text in Train exists in Test."""
    print("--- 1. ANALYZING DATA LEAKAGE (TRAIN vs TEST) ---")
    train_hashes = set()

    print("Hashing Train files...")
    for author in os.listdir(train_dir):
        author_path = os.path.join(train_dir, author)
        if not os.path.isdir(author_path): continue
        for file in os.listdir(author_path):
            file_hash = get_file_hash(os.path.join(author_path, file))
            if file_hash: train_hashes.add(file_hash)

    print("Checking Test files against Train hashes...")
    overlap_count = 0
    for author in os.listdir(test_dir):
        author_path = os.path.join(test_dir, author)
        if not os.path.isdir(author_path): continue
        for file in os.listdir(author_path):
            file_hash = get_file_hash(os.path.join(author_path, file))
            if file_hash in train_hashes:
                overlap_count += 1

    if overlap_count == 0:
        print("SUCCESS: 0 overlapping texts found. Train and Test are completely disjoint.")
    else:
        print(f"WARNING: Found {overlap_count} overlapping texts between Train and Test!")
    print("-" * 50)


def create_validation_split(train_dir, val_split_size=5):
    """Moves 'val_split_size' texts per author from Train to a new Val folder."""
    print(f"\n--- 2. CREATING VALIDATION SPLIT ({val_split_size} texts per author) ---")

    base_dataset_dir = os.path.dirname(train_dir)
    val_dir = os.path.join(base_dataset_dir, "C50val")

    if os.path.exists(val_dir):
        print(f"⚠alidation directory already exists at: {val_dir}")
        print("To prevent accidentally deleting more training data, the script will abort the split.")
        print("If you want to re-run, delete the C50val folder first.")
        return val_dir

    os.makedirs(val_dir)
    print(f"Created validation directory: {val_dir}")

    authors = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    total_moved = 0

    for author in authors:
        train_author_dir = os.path.join(train_dir, author)
        val_author_dir = os.path.join(val_dir, author)

        # Create author folder in Val
        os.makedirs(val_author_dir, exist_ok=True)

        # Get all text files for this author
        files = [f for f in os.listdir(train_author_dir) if f.endswith(".txt")]

        # Randomly shuffle your data before splitting to ensure that each subset is representative.
        random.seed(42)
        random.shuffle(files)

        # Select the texts to move
        val_files = files[:val_split_size]

        # MOVE the files
        for file in val_files:
            src_path = os.path.join(train_author_dir, file)
            dest_path = os.path.join(val_author_dir, file)
            shutil.move(src_path, dest_path)
            total_moved += 1

    print(f"SUCCESS: Moved {total_moved} files ({val_split_size} per author) to C50val.")
    print("Training set now has 45 texts per author, and validation has 5.")
    return val_dir


if __name__ == "__main__":
    train_dir = config.REUTERS_TRAIN_DIR
    test_dir = config.REUTERS_TEST_DIR

    analyze_overlap(train_dir, test_dir)
    create_validation_split(train_dir, val_split_size=5)