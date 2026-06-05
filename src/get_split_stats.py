#import json
#import os
#import sys
#from collections import defaultdict
#
## Add the parent directory to sys.path so we can import config.py
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#
## Import the exact paths from your config file!
#from config import DARK_REDDIT_TRAIN, DARK_REDDIT_VAL, DARK_REDDIT_TEST
#
#files = {
#    "Train": DARK_REDDIT_TRAIN,
#    "Val": DARK_REDDIT_VAL,
#    "Test": DARK_REDDIT_TEST
#}
#
#global_texts = 0
#global_words = 0
#global_chars = 0
#global_authors = set()
#
#split_stats = {
#    "Train": {"texts": 0, "words": 0},
#    "Val": {"texts": 0, "words": 0},
#    "Test": {"texts": 0, "words": 0}
#}
#
## Dictionary to hold stats for each author
#author_stats = defaultdict(lambda: {
#    "texts": 0, "Train": 0, "Val": 0, "Test": 0,
#    "words": 0, "min_words": float('inf'), "max_words": 0
#})
#
#print("Calculating dataset statistics across splits...\n")
#
#for split_name, file_path in files.items():
#    if not os.path.exists(file_path):
#        print(f"Warning: Could not find {file_path}. Skipping {split_name} split.")
#        continue
#
#    with open(file_path, 'r', encoding='utf-8') as f:
#        for line in f:
#            if not line.strip():
#                continue
#
#            data = json.loads(line)
#            author = data.get("author", "unknown")
#            text = data.get("comment", "")
#
#            words = len(text.split())
#            chars = len(text)
#
#            # Update Global
#            global_texts += 1
#            global_words += words
#            global_chars += chars
#            global_authors.add(author)
#
#            # Update Split
#            split_stats[split_name]["texts"] += 1
#            split_stats[split_name]["words"] += words
#
#            # Update Author
#            astats = author_stats[author]
#            astats["texts"] += 1
#            astats[split_name] += 1
#            astats["words"] += words
#            if words < astats["min_words"]:
#                astats["min_words"] = words
#            if words > astats["max_words"]:
#                astats["max_words"] = words
#
## --- PRINT OUTPUT ---
#print("=========================================")
#print("       GLOBAL DATASET STATISTICS         ")
#print("=========================================")
#print(f"Total Texts:      {global_texts:,}")
#print(f"Total Words:      {global_words:,}")
#print(f"Total Characters: {global_chars:,}")
#print(f"Total Authors:    {len(global_authors):,}")
#if global_texts > 0:
#    print(f"Avg Text Length:  {global_words / global_texts:.1f} words ({global_chars / global_texts:.1f} chars)")
#
#print("\n=========================================")
#print("            DATA SPLIT STATS             ")
#print("=========================================")
#for split_name in ["Train", "Val", "Test"]:
#    s_texts = split_stats[split_name]["texts"]
#    if global_texts > 0 and s_texts > 0:
#        percent = (s_texts / global_texts) * 100
#        print(f"{split_name:5} | Texts: {s_texts:,} ({percent:.1f}%) | Words: {split_stats[split_name]['words']:,}")
#
#print("\n=========================================")
#print("        AUTHOR-BY-AUTHOR BREAKDOWN       ")
#print("=========================================")
## Sort authors by total number of texts (highest first)
#for author, astats in sorted(author_stats.items(), key=lambda x: x[1]["texts"], reverse=True):
#    percent = (astats["texts"] / global_texts) * 100
#    avg_words = astats["words"] / astats["texts"]
#
#    # Handle the edge case where an author has 0 words somehow
#    min_w = astats["min_words"] if astats["min_words"] != float('inf') else 0
#
#   print(f"Alias: {author}")
#    print(f"  Total Texts : {astats['texts']:,} ({percent:.1f}%)")
#    print(f"  Split (T/V/T): {astats['Train']} / {astats['Val']} / {astats['Test']}")
#    print(f"  Avg Words   : {avg_words:.1f}")
#    print(f"  Length Range: {min_w} - {astats['max_words']} words\n")









# ---------------------------------------------------------------------------

import os
import sys
from collections import defaultdict

# Add the parent directory to sys.path so we can import config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the exact Reuters paths from your config file!
from config import REUTERS_TRAIN_DIR, REUTERS_VAL_DIR, REUTERS_TEST_DIR

split_dirs = {
    "Train": REUTERS_TRAIN_DIR,
    "Val": REUTERS_VAL_DIR,
    "Test": REUTERS_TEST_DIR
}

global_texts = 0
global_words = 0
global_chars = 0
global_authors = set()

split_stats = {
    "Train": {"texts": 0, "words": 0},
    "Val": {"texts": 0, "words": 0},
    "Test": {"texts": 0, "words": 0}
}

# Dictionary to hold stats for each author
author_stats = defaultdict(lambda: {
    "texts": 0, "Train": 0, "Val": 0, "Test": 0,
    "words": 0, "min_words": float('inf'), "max_words": 0
})

print("Calculating Reuters dataset statistics across splits...\n")

for split_name, split_dir in split_dirs.items():
    if not os.path.exists(split_dir):
        print(f"Warning: Could not find {split_dir}. Skipping {split_name} split.")
        continue

    # Iterate through each author folder (e.g., AaronPressman)
    for author_name in os.listdir(split_dir):
        author_path = os.path.join(split_dir, author_name)

        # Skip if it's not a directory
        if not os.path.isdir(author_path):
            continue

        # Iterate through each .txt file inside the author's folder
        for filename in os.listdir(author_path):
            if not filename.endswith(".txt"):
                continue

            file_path = os.path.join(author_path, filename)

            # Using errors='ignore' because Reuters is an older dataset and might have weird encodings
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            words = len(text.split())
            chars = len(text)

            # Update Global
            global_texts += 1
            global_words += words
            global_chars += chars
            global_authors.add(author_name)

            # Update Split
            split_stats[split_name]["texts"] += 1
            split_stats[split_name]["words"] += words

            # Update Author
            astats = author_stats[author_name]
            astats["texts"] += 1
            astats[split_name] += 1
            astats["words"] += words
            if words < astats["min_words"]:
                astats["min_words"] = words
            if words > astats["max_words"]:
                astats["max_words"] = words

# --- PRINT OUTPUT ---
print("=========================================")
print("       GLOBAL DATASET STATISTICS         ")
print("=========================================")
print(f"Total Texts:      {global_texts:,}")
print(f"Total Words:      {global_words:,}")
print(f"Total Characters: {global_chars:,}")
print(f"Total Authors:    {len(global_authors):,}")
if global_texts > 0:
    print(f"Avg Text Length:  {global_words / global_texts:.1f} words ({global_chars / global_texts:.1f} chars)")

print("\n=========================================")
print("            DATA SPLIT STATS             ")
print("=========================================")
for split_name in ["Train", "Val", "Test"]:
    s_texts = split_stats[split_name]["texts"]
    if global_texts > 0 and s_texts > 0:
        percent = (s_texts / global_texts) * 100
        print(f"{split_name:5} | Texts: {s_texts:,} ({percent:.1f}%) | Words: {split_stats[split_name]['words']:,}")

print("\n=========================================")
print("        AUTHOR-BY-AUTHOR BREAKDOWN       ")
print("=========================================")
# Sort authors alphabetically for Reuters since there are 50 of them and they should all be uniform
for author, astats in sorted(author_stats.items()):
    percent = (astats["texts"] / global_texts) * 100
    avg_words = astats["words"] / astats["texts"]

    min_w = astats["min_words"] if astats["min_words"] != float('inf') else 0

    print(f"Author: {author}")
    print(f"  Total Texts : {astats['texts']:,} ({percent:.1f}%)")
    print(f"  Split (T/V/T): {astats['Train']} / {astats['Val']} / {astats['Test']}")
    print(f"  Avg Words   : {avg_words:.1f}")
    print(f"  Length Range: {min_w} - {astats['max_words']} words\n")





