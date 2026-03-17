import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import Dataset

import config
from src.training.load_train_data import load_train_val_data
from src.training.hyper_and_trainer import get_hyperparameters, AuthorTripletTrainer


def main():
    # 1. Get Hyperparameters
    args = get_hyperparameters()

    print(f"========================================")
    print(f"   STARTING METRIC LEARNING (TRIPLET LOSS)")
    print(f"   Model:     {args.model}")
    print(f"   Dataset:   {args.dataset}")
    print(f"   Pooling:   {args.pooling.upper()}")
    print(f"   Suffix:    {args.suffix if args.suffix else 'None'}")
    print(f"   Batch Size:{args.batch_size} (Crucial for Triplet Mining)")
    print(f"========================================")

    # ---> NEW: Dynamic Save Directory based on Suffix <---
    adapter_name = f"{args.model}_{args.dataset}{args.suffix}"
    adapter_dir = os.path.join(config.RESULTS_DIR, "adapters", adapter_name)
    # ----------------------------------------------------

    # 2. Load Train and Validation Data (Strictly from files)
    train_texts, val_texts, train_labels, val_labels = load_train_val_data(args.dataset)

    if "e5" in args.model.lower():
        print("   [Data] Prepending 'passage: ' prefix for E5 model...")
        train_texts = [f"passage: {t}" for t in train_texts]
        val_texts = [f"passage: {t}" for t in val_texts]

    # Convert string labels to integer IDs
    unique_authors = sorted(list(set(train_labels + val_labels)))
    author_to_id = {author: idx for idx, author in enumerate(unique_authors)}

    train_ids = [author_to_id[a] for a in train_labels]
    val_ids = [author_to_id[a] for a in val_labels]

    train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_ids})
    val_dataset = Dataset.from_dict({"text": val_texts, "labels": val_ids})

    # 3. Model ID Selection
    model_alias = args.model.lower()
    is_decoder = "llama" in model_alias

    if "e5_small" in model_alias:
        model_id = config.E5_SMALL_ID
    elif "e5_large" in model_alias:
        model_id = config.E5_LARGE_ID
    elif "llama2" in model_alias:
        model_id = config.LLAMA2_CHECKPOINT_DIR
    elif "llama3" in model_alias:
        model_id = getattr(config, 'LLAMA3_CHECKPOINT_DIR', "meta-llama/Meta-Llama-3.1-8B")
    else:
        raise ValueError(f"Unknown model: {model_alias}")

    print(f"   [Model] Loading Base: {model_id}")

    # 4. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # 5. Load Model & Configure LoRA
    if is_decoder:
        print("   [Config] Detected DECODER architecture (Llama).")

        model = AutoModel.from_pretrained(
            model_id,
            quantization_config=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=args.r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias=args.bias,
            use_dora=args.use_dora,
        )

    else:
        print("   [Config] Detected ENCODER architecture (E5/BERT).")
        model = AutoModel.from_pretrained(model_id, device_map="auto")

        target_modules = ["query", "key", "value", "dense", "intermediate.dense", "output.dense"]

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=args.r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias=args.bias,
            use_dora=args.use_dora,
        )

    model = get_peft_model(model, peft_config)
    print("\n   [LoRA Config] Trainable Parameters:")
    model.print_trainable_parameters()

    # 6. Tokenization
    def tokenize_function(examples):
        # Splitting documents dynamically during mapping breaks the sampler logic.
        # Therefore, during *training*, enforce truncation
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 7. Training Args
    training_args = TrainingArguments(
        output_dir=adapter_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,

        report_to="wandb",
        run_name=f"TRIPLET-{model_alias}-{args.dataset}{args.suffix}",
        remove_unused_columns=False,
        lr_scheduler_type=args.lr_scheduler,
    )

    # 8. Instantiate the Custom Trainer
    trainer = AuthorTripletTrainer(
        triplet_margin=args.triplet_margin,
        pooling=args.pooling, 
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    print("   [Train] Starting training loop...")
    trainer.train()

    print(f"   [Save] Saving BEST adapter to {adapter_dir}")
    model.save_pretrained(adapter_dir)


if __name__ == "__main__":
    main()