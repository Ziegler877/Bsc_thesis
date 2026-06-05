import torch

# 1. Put your exact file path here
file_path = r"D:\12._Semester\results_final\embeddings\e5_large_darkreddit_lora_run1.pt"

print(f"Loading {file_path}...")

# 2. Load the file (map_location='cpu' prevents errors if you saved it on a GPU)
data = torch.load(file_path, map_location='cpu')

# 3. Check what kind of object it is
print(f"\nData Type: {type(data)}")

# 4. If it's a dictionary, print the keys
if isinstance(data, dict):
    print("Keys inside the file:")
    for key in data.keys():
        # Print the key and the type/size of the data inside it
        item = data[key]
        if isinstance(item, torch.Tensor):
            print(f" - '{key}': Tensor of shape {item.shape}")
        elif isinstance(item, list):
            print(f" - '{key}': List of length {len(item)}")
        else:
            print(f" - '{key}': {type(item)}")

# 5. If it's just a raw tensor, print its shape
elif isinstance(data, torch.Tensor):
    print(f"Shape of the Tensor: {data.shape}")
    print("WARNING: This is just a raw tensor. There are no labels saved in this file!")

else:
    print("This is an unknown format.")