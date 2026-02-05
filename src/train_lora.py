import os
import sys
import argparse
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import Dataset

import config
from src import data_loader


def run_lora_training(model_alias, dataset_alias, epochs=3, batch_size=4):
    print(f"========================================")
    print(f"   STARTING LORA TRAINING (AUTHORSHIP SPECIALIZED)")
    print(f"   Model:   {model_alias}")
    print(f"   Dataset: {dataset_alias}")
    print(f"========================================")

    adapter_dir = os.path.join(config.RESULTS_DIR, "adapters", f"{model_alias}_{dataset_alias}")

    # 1. Load Data
    print("   [Data] Loading dataset...")
    train_texts, _ = data_loader.load_dataset(dataset_alias, "train")

    # Optional: Add a "Style" prefix to help the model distinguish this task
    # (Only use this if you also add it in main.py evaluation!)
    # train_texts = [f"Analyze style: {t}" for t in train_texts]

    dataset = Dataset.from_dict({"text": train_texts})

    # 2. Config & ID Selection
    model_alias = model_alias.lower()
    is_decoder = "llama" in model_alias

    if "e5_small" in model_alias:
        model_id = "intfloat/e5-small-v2"
    elif "e5_large" in model_alias:
        model_id = "intfloat/e5-large-v2"
    elif "llama2" in model_alias:
        model_id = config.LLAMA2_CHECKPOINT_DIR
    elif "llama4" in model_alias:
        # Fallback if config doesn't have it
        model_id = getattr(config, 'LLAMA4_CHECKPOINT_DIR', "meta-llama/Llama-4-Maverick-17B-128E-Instruct")
    else:
        raise ValueError(f"Unknown model: {model_alias}")

    print(f"   [Model] Loading Base: {model_id}")

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Load Model & Configure LoRA
    if is_decoder:
        # --- DECODER (LLAMA) SETUP ---

        # Enable QLoRA (4-bit) if you are on the 17B model to save memory,
        # or use standard loading if you have massive GPUs.
        # Assuming you want to be safe with memory:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)

        # CRITICAL FIX: Target ALL linear layers, not just q/v.
        # This allows the model to adapt its internal logic (MLP), not just attention.
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,  # Increased from 8 -> 32 (Better capacity for style)
            lora_alpha=64,  # 2x Rank
            lora_dropout=0.1,  # Increased from 0.05 (Prevents keyword overfitting)
            target_modules=target_modules,
            bias="none"
        )

        def tokenize_function(examples):
            # Llama doesn't need "max_length" padding for training usually,
            # but for safety/consistency we keep it.
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    else:
        # --- ENCODER (E5/BERT) SETUP ---
        model = AutoModelForMaskedLM.from_pretrained(model_id, device_map="auto")

        # Target all linear layers for BERT-like models too
        target_modules = ["query", "key", "value", "dense"]

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=32,  # Higher rank
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=target_modules
        )

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    # 5. Apply LoRA
    model = get_peft_model(model, peft_config)
    print("\n   [LoRA Config] Trainable Parameters:")
    model.print_trainable_parameters()

    # 6. Tokenize
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 7. Training Args
    training_args = TrainingArguments(
        output_dir=adapter_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Simulates larger batch size (more stable gradients)
        num_train_epochs=epochs,
        learning_rate=1e-4,  # Lowered from 2e-4 (More gentle updates)
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="wandb",
        run_name=f"TRAIN-{model_alias}-{dataset_alias}",
        warmup_ratio=0.03,  # Small warmup helps stability
        weight_decay=0.01  # regularization
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=not is_decoder),
    )

    print("   [Train] Starting training loop...")
    trainer.train()

    print(f"   [Save] Saving adapter to {adapter_dir}")
    model.save_pretrained(adapter_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)  # Keep low for 17B
    args = parser.parse_args()

    run_lora_training(args.model, args.dataset, args.epochs, args.batch_size)