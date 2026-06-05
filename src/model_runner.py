import torch
import torch.nn as nn
import torch.nn.functional as F
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
#  HELPER: Generalized Mean Pooling (GeM)
# ==========================================
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x, attention_mask=None):
        # Optimization: Use scalar 'p' for calculation if possible to speed up inference
        p_val = self.p.item() if not self.p.requires_grad else self.p

        if attention_mask is not None:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
            x = x * input_mask_expanded

            # Clamp to avoid 0s or negatives (ReLU-like behavior is standard for GeM)
            x = x.clamp(min=self.eps)

            # Sum of x^p
            sum_pow = torch.sum(x.pow(p_val), dim=1)

            # Count non-padded tokens
            count = input_mask_expanded.sum(dim=1)
            count = count.clamp(min=self.eps)

            # Average and root
            return (sum_pow / count).pow(1.0 / p_val)
        else:
            return F.avg_pool1d(x.clamp(min=self.eps).pow(p_val), (x.size(-1))).pow(1. / p_val)


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
    def __init__(self, model_alias, adapter_path=None, pooling_type="mean", use_chunking=False):
        self.model_alias = model_alias
        self.device = config.DEVICE
        self.pooling_type = pooling_type
        self.use_chunking = use_chunking

        # 1. Determine ID from Config
        if "small" in model_alias:
            self.model_id = config.E5_SMALL_ID
        else:
            self.model_id = config.E5_LARGE_ID

        print(
            f"   [{model_alias}] Loading E5 (Encoder) from: {self.model_id} | Pool: {pooling_type} | Chunk: {use_chunking}")

        # 2. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # 3. Load Model
        self.model = AutoModel.from_pretrained(self.model_id, device_map="auto")

        # 4. Apply LoRA (With Fix)
        self.model = load_adapter_safe(self.model, adapter_path, model_alias)

        # 5. Initialize GeM if selected
        if self.pooling_type == "gmp":
            self.gem = GeM().to(self.device)

        self.model.eval()

    def get_embeddings(self, text_list, batch_size=16):
        if self.use_chunking:
            return self._get_embeddings_chunked(text_list)
        else:
            return self._get_embeddings_truncated(text_list, batch_size)

    def _get_embeddings_truncated(self, text_list, batch_size):
        all_embeddings = []
        for i in tqdm(range(0, len(text_list), batch_size), desc=f"   [{self.model_alias}] Truncated"):
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

            # Select Pooling Strategy
            if self.pooling_type == "gmp":
                # Use GeM Layer
                embeddings = self.gem(last_hidden, inputs['attention_mask'])
            elif self.pooling_type == "dynamic":
                # FORCED LAST-TOKEN POOLING FOR E5 (For empirical baseline testing)
                sequence_lengths = inputs['attention_mask'].sum(dim=1) - 1
                batch_size_actual = last_hidden.shape[0]
                embeddings = last_hidden[torch.arange(batch_size_actual, device=self.device), sequence_lengths]
            else:
                # Use Standard Mean Pooling (also defaults here for dynamic since E5 is an encoder)
                embeddings = self._mean_pooling(last_hidden, inputs['attention_mask'])

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def _get_embeddings_chunked(self, text_list):
        all_embeddings = []
        # Process document by document for chunking logic
        for text in tqdm(text_list, desc=f"   [{self.model_alias}] Chunking"):
            # 1. Tokenize full text (no truncation yet)
            tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=False)
            input_ids = tokens['input_ids'][0]  # shape [seq_len]

            # 2. Split into 512-token chunks
            chunk_size = 512
            chunks = []
            for i in range(0, len(input_ids), chunk_size):
                chunk_ids = input_ids[i:i + chunk_size]
                chunks.append(chunk_ids)

            # 3. Embed chunks
            chunk_vecs = []
            for c_ids in chunks:
                c_ids = c_ids.unsqueeze(0).to(self.device)  # [1, seq_len]
                attention_mask = torch.ones_like(c_ids).to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids=c_ids, attention_mask=attention_mask)

                last_hidden = outputs.last_hidden_state
                if self.pooling_type == "gmp":
                    vec = self.gem(last_hidden, attention_mask)
                elif self.pooling_type == "dynamic":
                    # FORCED LAST-TOKEN POOLING FOR E5 (For empirical baseline testing)
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    batch_size_actual = last_hidden.shape[0]
                    vec = last_hidden[torch.arange(batch_size_actual, device=self.device), sequence_lengths]
                else:
                    vec = self._mean_pooling(last_hidden, attention_mask)
                chunk_vecs.append(vec.cpu())

            # 4. Average Chunk Vectors to get Document Vector
            chunk_vecs = torch.cat(chunk_vecs, dim=0)  # [num_chunks, hidden_dim]
            doc_vec = torch.mean(chunk_vecs, dim=0, keepdim=True)  # [1, hidden_dim]
            all_embeddings.append(doc_vec)

        return torch.cat(all_embeddings, dim=0)

    def _mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# ==========================================
#  CLASS 2: Llama Runner (The Decoder Specialist)
# ==========================================
class LlamaRunner:
    def __init__(self, model_alias, adapter_path=None, pooling_type="mean", use_chunking=False):
        self.model_alias = model_alias
        self.device = config.DEVICE
        self.pooling_type = pooling_type
        self.use_chunking = use_chunking

        # --- MODEL SELECTION (Updated for Llama 3) ---
        if "llama3" in model_alias:
            # Tries to find LLAMA3 in config, otherwise defaults to HF Hub ID
            self.model_id = getattr(config, 'LLAMA3_CHECKPOINT_DIR', "meta-llama/Meta-Llama-3.1-8B")
        elif "scout" in model_alias:
            self.model_id = config.LLAMA4_SCOUT_CHECKPOINT_DIR
        elif "llama4" in model_alias:
            self.model_id = config.LLAMA4_CHECKPOINT_DIR
        else:
            self.model_id = config.LLAMA2_CHECKPOINT_DIR

        print(
            f"   [{model_alias}] Loading Llama (Decoder) from: {self.model_id} | Pool: {pooling_type} | Chunk: {use_chunking}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fix padding side for dynamic pooling
        self.tokenizer.padding_side = "right"

        # Standard FP16 loading (No 4-bit quantization)
        self.model = AutoModel.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # 4. Apply LoRA
        self.model = load_adapter_safe(self.model, adapter_path, model_alias)

        # 5. Initialize GeM if selected
        if self.pooling_type == "gmp":
            self.gem = GeM().to(self.device)

        self.model.eval()

    def get_embeddings(self, text_list, batch_size=1):
        # Enforce small batch size for Scout/Llama if needed
        if batch_size > 4:
            batch_size = 1

        if self.use_chunking:
            return self._get_embeddings_chunked(text_list)
        else:
            return self._get_embeddings_truncated(text_list, batch_size)

    def _get_embeddings_truncated(self, text_list, batch_size):
        all_embeddings = []
        for i in tqdm(range(0, len(text_list), batch_size), desc=f"   [{self.model_alias}] Truncated"):
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

            # Select Pooling Strategy
            if self.pooling_type == "gmp":
                embeddings = self.gem(hidden_states, inputs['attention_mask'])
            elif self.pooling_type == "dynamic":
                sequence_lengths = inputs['attention_mask'].sum(dim=1) - 1
                batch_size_actual = hidden_states.shape[0]
                embeddings = hidden_states[torch.arange(batch_size_actual, device=self.device), sequence_lengths]
            else:
                embeddings = self._mean_pooling(hidden_states, inputs['attention_mask'])

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def _get_embeddings_chunked(self, text_list):
        all_embeddings = []
        for text in tqdm(text_list, desc=f"   [{self.model_alias}] Chunking"):
            # 1. Tokenize full
            tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=False)
            input_ids = tokens['input_ids'][0]

            # 2. Split
            chunk_size = 512
            chunks = []
            for i in range(0, len(input_ids), chunk_size):
                chunks.append(input_ids[i:i + chunk_size])

            # 3. Embed chunks
            chunk_vecs = []
            for c_ids in chunks:
                c_ids = c_ids.unsqueeze(0).to(self.device)
                attention_mask = torch.ones_like(c_ids).to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids=c_ids, attention_mask=attention_mask, output_hidden_states=True)

                hidden_states = outputs.hidden_states[-1]
                if self.pooling_type == "gmp":
                    vec = self.gem(hidden_states, attention_mask)
                elif self.pooling_type == "dynamic":
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    batch_size_actual = hidden_states.shape[0]
                    vec = hidden_states[torch.arange(batch_size_actual, device=self.device), sequence_lengths]
                else:
                    vec = self._mean_pooling(hidden_states, attention_mask)
                chunk_vecs.append(vec.cpu())

            # 4. Average Chunks
            chunk_vecs = torch.cat(chunk_vecs, dim=0)
            doc_vec = torch.mean(chunk_vecs, dim=0, keepdim=True)
            all_embeddings.append(doc_vec)

        return torch.cat(all_embeddings, dim=0)

    def _mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# ==========================================
#  MAIN ENTRY POINT
# ==========================================
def run_pipeline(model_alias, train_texts, test_texts, use_lora=False, dataset_alias="unknown", suffix="",
                 pooling="mean", chunking=False, subset_size=None):
    """
    Main function to load models and generate embeddings.
    Suffix allows finding specific LoRA folders like 'e5_small_darkreddit_expA_3ep'
    subset_size: If provided, indicates that data has been subsampled (log info only).
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

    # 2. Select Runner (THIS IS NOW FIXED AND OUTDENTED)
    if "e5" in model_alias:
        runner = E5Runner(model_alias, adapter_path=adapter_path, pooling_type=pooling, use_chunking=chunking)
        batch_size = 16
        print("   [Data] Prepending 'passage: ' prefix for E5 model evaluation...")
        train_texts = [f"passage: {t}" for t in train_texts]
        test_texts = [f"passage: {t}" for t in test_texts]

    else:
        runner = LlamaRunner(model_alias, adapter_path=adapter_path, pooling_type=pooling, use_chunking=chunking)
        batch_size = 4

    # 3. Generate
    mode_str = "CHUNKED" if chunking else "TRUNCATED"
    subset_str = f" | Subset: {subset_size}" if subset_size else ""

    print(f"   [Pipeline] Generating Train Embeddings (Pool: {pooling}, Mode: {mode_str}{subset_str})...")
    train_vecs = runner.get_embeddings(train_texts, batch_size=batch_size)

    print(f"   [Pipeline] Generating Test Embeddings (Pool: {pooling}, Mode: {mode_str}{subset_str})...")
    test_vecs = runner.get_embeddings(test_texts, batch_size=batch_size)

    return train_vecs, test_vecs