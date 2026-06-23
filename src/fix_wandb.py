import wandb

# 1. Setup project
ENTITY = "e11909868-tu-wien"
PROJECT = "BSC Thesis"

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

print(f"Found {len(runs)} runs. Scanning for missing metadata...")

count = 0
for run in runs:
    if run.name.startswith("TRAIN-") and "dataset" not in run.config:

        parts = run.name.split("-")

        if len(parts) >= 3:
            detected_model = parts[1]
            detected_dataset = parts[2]

            run.config["dataset"] = detected_dataset
            run.config["model"] = detected_model
            run.update()

            print(f" Fixed: {run.name} -> set dataset='{detected_dataset}'")
            count += 1

print(f"\nTotal runs fixed: {count}")