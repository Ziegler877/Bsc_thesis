import torch
import os
import sys

BASE_FILE = r"..\results\embeddings\e5_small_reuters.pt"
LORA_FILE = r"..\results\embeddings\e5_small_reuters_lora.pt"

print(f"--- VERGLEICHE EMBEDDINGS ---")
print(f"1. Base: {BASE_FILE}")
print(f"2. LoRA: {LORA_FILE}")

if not os.path.exists(BASE_FILE) or not os.path.exists(LORA_FILE):
    print("(!) Eine der Dateien wurde nicht gefunden. Bitte Pfade prüfen.")
    sys.exit()

# Laden
base_data = torch.load(BASE_FILE, map_location="cpu")
lora_data = torch.load(LORA_FILE, map_location="cpu")

vec_base = base_data['test_vecs']
vec_lora = lora_data['test_vecs']

if torch.equal(vec_base, vec_lora):
    print("\n ALARM: Die Embeddings sind IDENTISCH (Werte sind gleich).")
else:
    diff = (vec_base - vec_lora).abs().mean().item()
    print("\n ERFOLG: Die Embeddings sind UNTERSCHIEDLICH!")
    print(f"   Durchschnittliche Abweichung pro Wert: {diff:.6f}")
