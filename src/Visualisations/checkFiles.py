import torch

file_path = r"D:\12._Semester\results_final\embeddings\e5_large_darkreddit_lora_run1.pt"

print(f"Loading {file_path}...")

data = torch.load(file_path, map_location='cpu')

print(f"\nData Type: {type(data)}")

if isinstance(data, dict):
    print("Keys inside the file:")
    for key in data.keys():
        item = data[key]
        if isinstance(item, torch.Tensor):
            print(f" - '{key}': Tensor of shape {item.shape}")
        elif isinstance(item, list):
            print(f" - '{key}': List of length {len(item)}")
        else:
            print(f" - '{key}': {type(item)}")

elif isinstance(data, torch.Tensor):
    print(f"Shape of the Tensor: {data.shape}")
    print("WARNING: This is just a raw tensor. There are no labels saved in this file!")

else:
    print("This is an unknown format.")