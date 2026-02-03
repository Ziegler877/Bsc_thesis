import torch
import sys

print("------------------------------------------------")
print(f"Python Version:   {sys.version.split()[0]}")
print(f"PyTorch Version:  {torch.__version__}")
print("------------------------------------------------")

# 1. Check if PyTorch was built with CUDA support
# This tells us if you INSTALLED the right version
print(f"CUDA Build Used:  {torch.version.cuda}")

# 2. Check if PyTorch can actually SEE the GPU
# This tells us if your Drivers/Hardware are communicating
is_available = torch.cuda.is_available()
print(f"CUDA Available:   {is_available}")

if is_available:
    print(f"Device Count:     {torch.cuda.device_count()}")
    print(f"Current Device:   {torch.cuda.current_device()}")
    print(f"Device Name:      {torch.cuda.get_device_name(0)}")
else:
    print("(!) CUDA is NOT available to PyTorch.")

print("------------------------------------------------")