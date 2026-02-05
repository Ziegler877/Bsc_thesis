import sys
import os
import argparse
import torch
import wandb

# --- IMPORTS ---
import config
from src import data_loader
from src import model_runner
from src import evaluation


def main():
    # ---------------------------------------------------------
    # 1. SETUP ARGUMENT PARSER
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Run Authorship Attribution Experiment")

    # CHANGE 1: Replaced 'llama4' with 'llama4_scout'
    parser.add_argument("--model", type=str, required=True,
                        choices=["e5_small", "e5_large", "llama2", "llama4_scout"],
                        help="Which model architecture to use")

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["reuters", "darkreddit"],
                        help="Which dataset to load")

    parser.add_argument("--lora", action="store_true",
                        help="Enable LoRA fine-tuning/adaptation")

    parser.add_argument("--device", type=str, default=config.DEVICE,
                        help="Override config device")

    # Optional: Add a suffix to the filename (e.g. "_ep3") to stop overwriting
    parser.add_argument("--suffix", type=str, default="",
                        help="Optional string appended to the saved .pt filename")


    # Alias for compatibility with shell scripts
    parser.add_argument("--use_adapter", action="store_true", help="Alias for --lora")

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 2. CONFIGURE BASED ON ARGUMENTS
    # ---------------------------------------------------------

    wandb.init(
        project="BSC Thesis",  # Name of your project on the website
        name=f"EVAL-{args.model}-{args.dataset}",  # Name of this specific run
        config={
            "model": args.model,
            "dataset": args.dataset,
            "lora": args.lora
        }
    )

    model_alias = args.model
    dataset_alias = args.dataset

    # Combine flags: --lora OR --use_adapter both work
    use_lora = args.lora or args.use_adapter

    job_name = f"{model_alias}_{dataset_alias}"
    if use_lora:
        job_name += " (LoRA)"

    print(f"========================================")
    print(f"   STARTING JOB: {job_name}")
    print(f"   Device: {args.device}")
    print(f"========================================")

    # ---------------------------
    # STEP 1: LOAD DATA
    # ---------------------------
    try:
        train_txt, train_lbl = data_loader.load_dataset(dataset_alias, "train")
        test_txt, test_lbl = data_loader.load_dataset(dataset_alias, "test")
    except Exception as e:
        print(f"(!) Data Load Failed: {e}")
        sys.exit(1)

    if train_lbl:
        # Filter Test set to only include authors seen in Training
        known_authors = set(train_lbl)
        filtered = [(t, l) for t, l in zip(test_txt, test_lbl) if l in known_authors]

        if not filtered:
            print("(!) Error: No overlapping authors between Train and Test.")
            sys.exit(1)

        test_txt, test_lbl = zip(*filtered)
        test_txt, test_lbl = list(test_txt), list(test_lbl)
        print(f"   [Data] Final Test Size: {len(test_txt)} samples")
    else:
        print("(!) Error: Training set empty.")
        sys.exit(1)

    # ---------------------------
    # STEP 2: RUN MODEL
    # ---------------------------
    try:
        # Prepare arguments for the runner
        runner_kwargs = {
            "model_alias": model_alias,
            "train_texts": train_txt,
            "test_texts": test_txt,
            "use_lora": use_lora,
            "dataset_alias": dataset_alias
            # CHANGE 2: Explicitly pass 4-bit flag so Scout doesn't crash
        }

        train_vecs, test_vecs = model_runner.run_pipeline(**runner_kwargs)


    except Exception as e:
        print(f"(!) Model Execution Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if train_vecs is None:
        print("(!) Model returned no embeddings.")
        sys.exit(1)

    # ---------------------------
    # STEP 2.5: SAVE EMBEDDINGS (FIXED)
    # ---------------------------
    # Create filename: e.g. "llama4_scout_reuters_lora.pt"
    lora_tag = "_lora" if use_lora else ""
    filename = f"{model_alias}_{dataset_alias}{lora_tag}{args.suffix}.pt"

    # Use correct config variable
    save_dir = config.EMBEDDINGS_DIR

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    print(f"   [Save] Saving embeddings to: {save_path}")
    torch.save({
        "train_vecs": train_vecs.cpu(),
        "test_vecs": test_vecs.cpu(),
        "train_labels": train_lbl,
        "test_labels": test_lbl
    }, save_path)

    # ---------------------------
    # STEP 3: EVALUATE & LOG
    # ---------------------------
    metrics = evaluation.run_evaluation(
        train_vecs=train_vecs,
        test_vecs=test_vecs,
        train_labels=train_lbl,
        test_labels=test_lbl,
        model_name=model_alias,
        dataset_name=dataset_alias,
        device=args.device,
        extra_info=f"LoRA: {use_lora} | Suffix: {args.suffix}"
    )

    if metrics:
        wandb.log(metrics)

    print("\n========================================")
    print(f"   JOB COMPLETE: {job_name}")
    print("==========================================")

    wandb.finish()

if __name__ == "__main__":
    main()