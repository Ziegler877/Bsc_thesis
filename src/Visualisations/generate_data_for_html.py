import json
import subprocess
from pathlib import Path
import numpy as np
import torch
import umap

# 1. Set the folder and the list of files to process
EMBEDDING_DIR = Path(r"C:\Users\maxid\Universität\12._Semester\BSC\Programm\results\embeddings\dynamic")
#D:\12._Semester\BSC\Cluster\results_2\embeddings\embeddings\try something out

file_list = [
    #"e5_large_darkreddit.pt",
    #"e5_large_darkreddit_chunked.pt",
    #"e5_large_darkreddit_dynamic.pt",
    #"e5_large_darkreddit_sub5.pt",
    #"e5_large_reuters.pt",
    #"e5_large_reuters_chunked.pt",
    #"e5_large_reuters_dynamic.pt",
    #"e5_large_reuters_sub5.pt",
    #"e5_small_darkreddit.pt",
    #"e5_small_darkreddit_chunked.pt",
    "e5_small_darkreddit_dynamic.pt",
    #"e5_small_darkreddit_sub5.pt",
    #"e5_small_reuters.pt",
    #"e5_small_reuters_chunked.pt",
    #"e5_small_reuters_dynamic.pt",
    #"e5_small_reuters_sub5.pt",
    #"llama2_darkreddit.pt",
    #"llama2_darkreddit_chunked.pt",
    #"llama2_darkreddit_dynamic.pt",
    #"llama2_darkreddit_sub5.pt",
    #"llama2_reuters.pt",
    #"llama2_reuters_chunked.pt",
    #"llama2_reuters_dynamic.pt",
    #"llama2_reuters_sub5.pt",
    #"llama3_darkreddit.pt",
    #"llama3_darkreddit_chunked.pt",
    #"llama3_darkreddit_dynamic.pt",
    #"llama3_darkreddit_sub5.pt",
    #"llama3_reuters.pt",
    #"llama3_reuters_chunked.pt",
    #"llama3_reuters_dynamic.pt",
    #"llama3_reuters_sub5.pt"
]

# 2. Standard color palette and shapes
shapes = ['circle', 'square', 'diamond', 'cross', 'triangle-up']
hex_palette = [
    "#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231",
    "#911EB4", "#46F0F0", "#F032E6", "#BCF60C", "#FABEBE"
]

# 3. Loop through every file in the list
for filename in file_list:
    data_path = EMBEDDING_DIR / filename

    if not data_path.exists():
        print(f"\n[ERROR] Skipping {filename} - File not found!")
        continue

    print(f"\n========================================")
    print(f"Processing: {filename}")
    print(f"========================================")

    # Load data
    saved_data = torch.load(data_path, map_location='cpu', weights_only=False)
    encodings = saved_data['test_vecs'].numpy()
    labels = saved_data['test_labels']

    # Map colors AND shapes
    unique_authors = list(set(labels))
    color_map = {}
    shape_map = {}

    for i, author in enumerate(unique_authors):
        color_map[author] = hex_palette[i % len(hex_palette)]  # Cycles 10 colors
        shape_map[author] = shapes[(i // len(hex_palette)) % len(shapes)]  # Cycles shapes every 10 authors

    # Run UMAP
    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    reduced = reducer.fit_transform(encodings)

    # Build unique JSON for this model
    file_stem = data_path.stem  # e.g., "llama3_reuters_lora_run1"
    output_json = f"umap_data_{file_stem}.json"

    export_data = []
    for i in range(len(encodings)):
        export_data.append({
            "x": float(reduced[i][0]),
            "y": float(reduced[i][1]),
            "label": str(labels[i]),
            "color": color_map[labels[i]],
            "shape": shape_map[labels[i]],
            "filename": f"Test_Sample_{i}"
        })

    with open(output_json, "w") as f:
        json.dump(export_data, f, indent=2)

    # Call the HTML generator
    output_html = f"UMAP_{file_stem}.html"
    title_text = f"UMAP: {file_stem.replace('_', ' ').upper()}"

    print(f"Generating HTML -> {output_html}")
    subprocess.run([
        "python", "create_html.py",
        output_json,
        output_html,
        "--title", title_text,
        "--download-name", file_stem
    ])

print("\n✓ ALL FILES PROCESSED SUCCESSFULLY!")