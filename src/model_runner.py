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
        if attention_mask is not None:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
            x = x * input_mask_expanded
            x = x.clamp(min=self.eps)
            sum_pow = torch.sum(x.pow(self.p), dim=1)
            count = input_mask_expanded.sum(dim=1)
            count = count.clamp(min=self.eps)
            return (sum_pow / count).pow(1.0 / self.p)
        else:
            return F.avg_pool1d(x.clamp(min=self.eps).pow(self.p), (x.size(-1))).pow(1. / self.p)


# ==========================================
#  HELPER: Centralized LoRA Loader with KEY FIX
# ==========================================
def load_adapter_safe(model, adapter_path, model_alias):
    if not adapter_path:
        print(f"   [{model_alias}] Info: No adapter requested. Running BASE model.")
        return model

    print(f"   [{model_alias}] [DEBUG] Attempting to load LoRA from: {adapter_path}")

    if not os.path.exists(adapter_path):
        print(f"   [{model_alias}] [ERROR] Adapter path not found on disk.")
        return model

    try:
        model = PeftModel.from_pretrained(model, adapter_path)

        safe_path = os.path.join(adapter_path, "adapter_model.safetensors")
        bin_path = os.path.join(adapter_path, "adapter_model.bin")

        if os.path.exists(safe_path):
            state_dict = load_file(safe_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            print(f"   [{model_alias}] [ERROR] No weight file found (safetensors/bin).")
            return model

        new_state_dict = {}
        fixed_count = 0
        for k, v in state_dict.items():
            new_key = k
            if "bert.encoder" in k:
                new_key = k.replace("bert.encoder", "encoder")
                fixed_count += 1
            new_state_dict[new_key] = v

        if fixed_count > 0:
            print(f"   [{model_alias}] [FIX] Renamed {fixed_count} keys (removed .bert prefix) to match model.")
            result = set_peft_model_state_dict(model, new_state_dict)
        else:
            result = set_peft_model_state_dict(model, state_dict)

        if len(result.missing_keys) > 0:
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

        if "small" in model_alias:
            self.model_id = config.E5_SMALL_ID
        else:
            self.model_id = config.E5_LARGE_ID

        print(
            f"   [{model_alias}] Loading E5 (Encoder) from: {self.model_id} | Pool: {pooling_type} | Chunk: {use_chunking}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id, device_map="auto")
        self.model = load_adapter_safe(self.model, adapter_path, model_alias)

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
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            last_hidden = outputs.last_hidden_state
            if self.pooling_type == "gmp":
                embs = self.gem(last_hidden, inputs['attention_mask'])
            else:
                embs = self._mean_pooling(last_hidden, inputs['attention_mask'])
            all_embeddings.append(embs.cpu())
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
                # Pad if strictly necessary? Usually transformers handles variable length in a batch,
                # but here we just process one chunk at a time or batch chunks.
                # Simplest: Process chunk immediately.
                chunks.append(chunk_ids)

            # 3. Embed chunks
            chunk_vecs = []
            # Batch these chunks if there are many (optional optimization),
            # here we loop simply as texts are rarely 10k+ tokens.
            for c_ids in chunks:
                c_ids = c_ids.unsqueeze(0).to(self.device)  # [1, seq_len]
                attention_mask = torch.ones_like(c_ids).to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids=c_ids, attention_mask=attention_mask)

                last_hidden = outputs.last_hidden_state
                if self.pooling_type == "gmp":
                    vec = self.gem(last_hidden, attention_mask)
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

        if "scout" in model_alias:
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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        self.model = load_adapter_safe(self.model, adapter_path, model_alias)

        if self.pooling_type == "gmp":
            self.gem = GeM().to(self.device)

        self.model.eval()

    def get_embeddings(self, text_list, batch_size=4):
        if "scout" in self.model_alias and batch_size > 4:
            batch_size = 4

        if self.use_chunking:
            return self._get_embeddings_chunked(text_list)
        else:
            return self._get_embeddings_truncated(text_list, batch_size)

    def _get_embeddings_truncated(self, text_list, batch_size):
        all_embeddings = []
        for i in tqdm(range(0, len(text_list), batch_size), desc=f"   [{self.model_alias}] Truncated"):
            batch_texts = text_list[i: i + batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states[-1]
            if self.pooling_type == "gmp":
                embs = self.gem(hidden_states, inputs['attention_mask'])
            else:
                embs = self._mean_pooling(hidden_states, inputs['attention_mask'])
            all_embeddings.append(embs.cpu())
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
                 pooling="mean", chunking=False):
    # 1. Determine Adapter Path
    adapter_path = None
    if use_lora:
        folder_name = f"{model_alias}_{dataset_alias}{suffix}"
        adapter_path = os.path.join(config.ADAPTERS_DIR, folder_name)
        if not os.path.exists(adapter_path):
            print(f"   [Pipeline] Warning: Specific adapter {folder_name} not found.")

    # 2. Select Runner
    if "e5" in model_alias:
        runner = E5Runner(model_alias, adapter_path=adapter_path, pooling_type=pooling, use_chunking=chunking)
        batch_size = 16
    else:
        runner = LlamaRunner(model_alias, adapter_path=adapter_path, pooling_type=pooling, use_chunking=chunking)
        batch_size = 4

    # 3. Generate
    mode_str = "CHUNKED" if chunking else "TRUNCATED"
    print(f"   [Pipeline] Generating Train Embeddings (Pool: {pooling}, Mode: {mode_str})...")
    train_vecs = runner.get_embeddings(train_texts, batch_size=batch_size)

    print(f"   [Pipeline] Generating Test Embeddings (Pool: {pooling}, Mode: {mode_str})...")
    test_vecs = runner.get_embeddings(test_texts, batch_size=batch_size)

    return train_vecs, test_vecs