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

    parser.add_argument("--model", type=str, required=True,
                        choices=["e5_small", "e5_large", "llama2", "llama3", "llama4_scout"],
                        help="Which model architecture to use")

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["reuters", "darkreddit"],
                        help="Which dataset to load")

    parser.add_argument("--lora", action="store_true",
                        help="Enable LoRA fine-tuning/adaptation")

    # Metadata for W&B
    parser.add_argument("--epochs", type=int, default=0,
                        help="Metadata: How many epochs was the adapter trained? (0 = Base Model)")

    # Pooling Strategy
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "gmp"],
                        help="Aggregation strategy: 'mean' (standard) or 'gmp' (Generalized Mean Pooling)")

    # Chunking Option
    parser.add_argument("--chunking", action="store_true",
                        help="If set, splits long texts into 512-token chunks and averages them. If not set, truncates at 512.")

    # NEW: Subset Option (for faster experiments or robustness checks)
    parser.add_argument("--subset", type=int, default=None,
                        help="If set, limits the number of texts per author (e.g., 5).")

    parser.add_argument("--device", type=str, default=config.DEVICE,
                        help="Override config device")

    parser.add_argument("--suffix", type=str, default="",
                        help="Optional string appended to the saved .pt filename")

    parser.add_argument("--use_adapter", action="store_true", help="Alias for --lora")

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 2. CONFIGURE BASED ON ARGUMENTS
    # ---------------------------------------------------------

    # Construct a smart Run Name
    run_name = f"EVAL-{args.model}-{args.dataset}"

    if args.epochs > 0:
        run_name += f"-{args.epochs}ep"
    elif "_base" in args.suffix:
        run_name += "-base"

    # Add pooling to run name
    if args.pooling != "mean":
        run_name += f"-{args.pooling}"

    # Add chunking to run name
    if args.chunking:
        run_name += "-chunked"

    # Add subset to run name
    if args.subset:
        run_name += f"-sub{args.subset}"

    # Initialize W&B
    wandb.init(
        project="BSC Thesis",
        name=run_name,
        config={
            "model": args.model,
            "dataset": args.dataset,
            "lora": args.lora,
            "epochs": args.epochs,
            "pooling": args.pooling,
            "chunking": args.chunking,
            "subset": args.subset,  # <--- Track subset size in W&B
            "device": args.device
        }
    )

    model_alias = args.model
    dataset_alias = args.dataset
    use_lora = args.lora or args.use_adapter

    job_name = f"{model_alias}_{dataset_alias}"
    if use_lora:
        job_name += f" (LoRA {args.epochs}ep)"

    job_name += f" [{args.pooling.upper()}]"
    if args.chunking:
        job_name += " [CHUNKED]"
    if args.subset:
        job_name += f" [SUBSET: {args.subset}]"

    print(f"========================================")
    print(f"   STARTING JOB: {job_name}")
    print(f"   Device:   {args.device}")
    print(f"   Epochs:   {args.epochs}")
    print(f"   Pooling:  {args.pooling}")
    print(f"   Chunking: {args.chunking}")
    print(f"   Subset:   {args.subset}")
    print(f"========================================")

    # ---------------------------
    # STEP 1: LOAD DATA
    # ---------------------------
    try:
        # Pass subset_size to loader
        train_txt, train_lbl = data_loader.load_dataset(dataset_alias, "train", subset_size=args.subset)
        test_txt, test_lbl = data_loader.load_dataset(dataset_alias,
                                                      "test")  # Usually don't subset test set, but can be added if needed
    except Exception as e:
        print(f"(!) Data Load Failed: {e}")
        sys.exit(1)

    if train_lbl:
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
        runner_kwargs = {
            "model_alias": model_alias,
            "train_texts": train_txt,
            "test_texts": test_txt,
            "use_lora": use_lora,
            "dataset_alias": dataset_alias,
            "suffix": args.suffix,
            "pooling": args.pooling,
            "chunking": args.chunking,
            "subset_size": args.subset  # Pass subset info for logging inside runner
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
    # STEP 2.5: SAVE EMBEDDINGS
    # ---------------------------
    lora_tag = "_lora" if use_lora else ""

    # If explicit epochs are given, auto-generate suffix if one wasn't provided
    final_suffix = args.suffix
    if args.epochs > 0 and not final_suffix:
        final_suffix = f"_{args.epochs}ep"

    # Add tags to filename
    if args.pooling == "gmp":
        final_suffix += "_gmp"
    if args.chunking:
        final_suffix += "_chunked"
    if args.subset:
        final_suffix += f"_sub{args.subset}"

    filename = f"{model_alias}_{dataset_alias}{lora_tag}{final_suffix}.pt"

    save_dir = config.EMBEDDINGS_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    print(f"   [Save] Saving embeddings to: {save_path}")
    torch.save({
        "train_vecs": train_vecs.cpu(),
        "test_vecs": test_vecs.cpu(),
        "train_labels": train_lbl,
        "test_labels": test_lbl,
        "epochs": args.epochs,
        "pooling": args.pooling,
        "chunking": args.chunking,
        "subset": args.subset
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
        extra_info=f"LoRA: {use_lora} | Ep: {args.epochs} | Pool: {args.pooling} | Chunk: {args.chunking} | Sub: {args.subset}",
        pooling=args.pooling,
        chunking=args.chunking,
        subset_size=args.subset  # Pass subset to evaluation
    )

    if metrics:
        wandb.log(metrics)

    print("\n========================================")
    print(f"   JOB COMPLETE: {job_name}")
    print("==========================================")

    wandb.finish()


if __name__ == "__main__":
    main()