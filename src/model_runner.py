import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import sys
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ==========================================
#  HELPER: Centralized LoRA Loader with KEY FIX
# ==========================================
def load_adapter_safe(model, adapter_path, model_alias):
    """
    Loads LoRA adapters and fixes Key Mismatches (e.g. .bert.encoder vs .encoder).
    """
    if not adapter_path:
        print(f"   [{model_alias}] Info: No adapter requested. Running BASE model.")
        return model

    print(f"   [{model_alias}] [DEBUG] Attempting to load LoRA from: {adapter_path}")

    if not os.path.exists(adapter_path):
        print(f"   [{model_alias}] [ERROR] Adapter path not found on disk.")
        return model

    try:
        # 1. Initialize the PEFT wrapper (this creates the layers with random weights)
        model = PeftModel.from_pretrained(model, adapter_path)

        # 2. Check if we need to manually fix keys
        safe_path = os.path.join(adapter_path, "adapter_model.safetensors")
        bin_path = os.path.join(adapter_path, "adapter_model.bin")

        if os.path.exists(safe_path):
            state_dict = load_file(safe_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            print(f"   [{model_alias}] [ERROR] No weight file found (safetensors/bin).")
            return model

        # 3. REPAIR KEYS (The Magic Fix)
        # The log showed the file has '.bert.encoder' but model wants '.encoder'
        new_state_dict = {}
        fixed_count = 0
        for k, v in state_dict.items():
            new_key = k

            # FIX FOR E5 (Bert mismatch)
            if "bert.encoder" in k:
                new_key = k.replace("bert.encoder", "encoder")
                fixed_count += 1

            new_state_dict[new_key] = v

        if fixed_count > 0:
            print(f"   [{model_alias}] [FIX] Renamed {fixed_count} keys (removed .bert prefix) to match model.")
            # 4. Load the fixed weights into the model
            result = set_peft_model_state_dict(model, new_state_dict)
        else:
            # If no keys needed fixing, the initial from_pretrained load likely worked,
            # but usually PeftModel.from_pretrained handles loading too.
            # We explicitly set dict here just to be safe if keys match perfectly.
            result = set_peft_model_state_dict(model, state_dict)

        # Check result
        if len(result.missing_keys) > 0:
            # Filter out non-lora missing keys to see if it's a real problem
            real_missing = [k for k in result.missing_keys if "lora" in k]
            if real_missing:
                print(f"   [{model_alias}] [WARN] Still missing LoRA keys: {real_missing[:3]}...")
            else:
                print(f"   [{model_alias}] [SUCCESS] LoRA Adapter active (Keys matched).")
        else:
            print(f"   [{model_alias}] [SUCCESS] LoRA Adapter active (Perfect match).")

    except Exception as e:
        print(f"   [{model_alias}] [ERROR] Failed to load adapter: {e}")

    return model


# ==========================================
#  CLASS 1: E5 Runner (The Encoder Specialist)
# ==========================================
class E5Runner:
    def __init__(self, model_alias, adapter_path=None):
        self.model_alias = model_alias
        self.device = config.DEVICE

        # 1. Determine ID from Config
        if "small" in model_alias:
            self.model_id = config.E5_SMALL_ID
        else:
            self.model_id = config.E5_LARGE_ID

        print(f"   [{model_alias}] Loading E5 (Encoder) from: {self.model_id}")

        # 2. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # 3. Load Model
        self.model = AutoModel.from_pretrained(self.model_id, device_map="auto")

        # 4. Apply LoRA (With Fix)
        self.model = load_adapter_safe(self.model, adapter_path, model_alias)

        self.model.eval()

    def get_embeddings(self, text_list, batch_size=16):
        all_embeddings = []

        for i in tqdm(range(0, len(text_list), batch_size), desc=f"   [{self.model_alias}]"):
            batch_texts = text_list[i: i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            last_hidden = outputs.last_hidden_state
            # Create embedding via mean pooling
            embeddings = self._mean_pooling(last_hidden, inputs['attention_mask'])
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def _mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# ==========================================
#  CLASS 2: Llama Runner (The Decoder Specialist)
# ==========================================
class LlamaRunner:
    def __init__(self, model_alias, adapter_path=None):
        self.model_alias = model_alias
        self.device = config.DEVICE

        if "scout" in model_alias:
            self.model_id = config.LLAMA4_SCOUT_CHECKPOINT_DIR
        elif "llama4" in model_alias:
            self.model_id = config.LLAMA4_CHECKPOINT_DIR
        else:
            self.model_id = config.LLAMA2_CHECKPOINT_DIR

        print(f"   [{model_alias}] Loading Llama (Decoder) from: {self.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Standard FP16 loading (No 4-bit quantization)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # 4. Apply LoRA
        self.model = load_adapter_safe(self.model, adapter_path, model_alias)

        self.model.eval()

    def get_embeddings(self, text_list, batch_size=4):
        # Enforce small batch size for Scout if needed
        if "scout" in self.model_alias and batch_size > 4:
            batch_size = 4

        all_embeddings = []
        for i in tqdm(range(0, len(text_list), batch_size), desc=f"   [{self.model_alias}]"):
            batch_texts = text_list[i: i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            # Use the last hidden state
            hidden_states = outputs.hidden_states[-1]
            embeddings = self._mean_pooling(hidden_states, inputs['attention_mask'])
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def _mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# ==========================================
#  MAIN ENTRY POINT
# ==========================================
def run_pipeline(model_alias, train_texts, test_texts, use_lora=False, dataset_alias="unknown", suffix=""):
    """
    Main function to load models and generate embeddings.
    Suffix allows finding specific LoRA folders like 'e5_small_darkreddit_expA_3ep'
    """

    # 1. Determine Adapter Path
    adapter_path = None
    if use_lora:
        # Construct path: e.g. "results/adapters/e5_small_darkreddit_expA_3ep"
        folder_name = f"{model_alias}_{dataset_alias}{suffix}"
        adapter_path = os.path.join(config.ADAPTERS_DIR, folder_name)

        # Fallback if specific suffix folder doesn't exist but generic does
        if not os.path.exists(adapter_path):
            print(f"   [Pipeline] Warning: Specific adapter {folder_name} not found.")
            # You might want to try without suffix here, or just fail gracefully
            # adapter_path = os.path.join(config.ADAPTERS_DIR, f"{model_alias}_{dataset_alias}")

    # 2. Select Runner
    if "e5" in model_alias:
        runner = E5Runner(model_alias, adapter_path=adapter_path)
        batch_size = 16
    else:
        runner = LlamaRunner(model_alias, adapter_path=adapter_path)
        batch_size = 4

    # 3. Generate
    print("   [Pipeline] Generating Train Embeddings...")
    train_vecs = runner.get_embeddings(train_texts, batch_size=batch_size)

    print("   [Pipeline] Generating Test Embeddings...")
    test_vecs = runner.get_embeddings(test_texts, batch_size=batch_size)

    return train_vecs, test_vecs