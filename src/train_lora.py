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
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

# Now these imports will work
import config
from src import data_loader


def run_lora_training(model_alias, dataset_alias, epochs=3, batch_size=4):
    print(f"========================================")
    print(f"   STARTING LORA TRAINING (STANDARD BASE)")
    print(f"   Model:   {model_alias}")
    print(f"   Dataset: {dataset_alias}")
    print(f"========================================")

    adapter_dir = os.path.join(config.RESULTS_DIR, "adapters", f"{model_alias}_{dataset_alias}")

    # 1. Load Data
    print("   [Data] Loading dataset...")
    train_texts, _ = data_loader.load_dataset(dataset_alias, "train")
    dataset = Dataset.from_dict({"text": train_texts})

    # 2. Config
    model_alias = model_alias.lower()
    is_decoder = "llama" in model_alias

    if "e5_small" in model_alias:
        model_id = "intfloat/e5-small-v2"
    elif "e5_large" in model_alias:
        model_id = "intfloat/e5-large-v2"
    elif "llama2" in model_alias:
        model_id = config.LLAMA2_CHECKPOINT_DIR
    elif "llama4" in model_alias:
        model_id = getattr(config, 'LLAMA4_CHECKPOINT_DIR', "meta-llama/Llama-4-Maverick-17B-128E-Instruct")
    else:
        raise ValueError(f"Unknown model: {model_alias}")

    print(f"   [Model] Loading Base: {model_id}")

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 4. Load Base Model (Standard FP16)
    # NO bitsandbytes, NO load_in_4bit
    if is_decoder:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,  # Standard Half-Precision
            trust_remote_code=True
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"]
        )

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    else:
        model = AutoModelForMaskedLM.from_pretrained(model_id, device_map="auto")
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["query", "value"]
        )

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    # 5. Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 6. Tokenize
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 7. Training Args
    training_args = TrainingArguments(
        output_dir=adapter_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="wandb",  # <--- Change "none" to "wandb"
        run_name=f"TRAIN-{model_alias}-{dataset_alias}",  # <--- Gives the run a nice name in the UI
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
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    run_lora_training(args.model, args.dataset, args.epochs, args.batch_size)