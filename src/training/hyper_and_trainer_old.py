import argparse
import random
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Sampler
from transformers import Trainer


def get_hyperparameters():
    """Parses and returns the hyperparameter arguments for training."""
    parser = argparse.ArgumentParser(description="LoRA Training with Triplet Loss")

    # Core settings
    parser.add_argument("--model", type=str, required=True, help="Alias: e5_small, e5_large, llama2, llama3")
    parser.add_argument("--dataset", type=str, required=True, help="Alias: reuters, darkreddit")

    # Early Stopping & Epochs
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=4, help="Stop after N epochs without improvement")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")

    # Batch Size (Will be automatically factored into P and K)
    parser.add_argument("--batch_size", type=int, default=4, help="E.g., 4 for Llama, 16 for E5")

    # LoRA settings
    parser.add_argument("--r", type=int, default=64, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA Alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--use_dora", action="store_true", help="Use DoRA instead of LoRA")
    parser.add_argument("--bias", type=str, default="none", choices=["none", "all", "lora_only"])

    # Optimizer settings
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning Rate")
    parser.add_argument("--lr_scheduler", type=str, default="linear")
    parser.add_argument("--triplet_margin", type=float, default=0.5, help="Margin for Triplet Loss")

    return parser.parse_args()


# ==========================================
#  CUSTOM P-K SAMPLER FOR TRIPLET LOSS
# ==========================================
class PKSampler(Sampler):
    """
    Ensures every batch contains P authors and K texts per author.
    Crucial for small batch sizes (like Llama) to guarantee valid triplets.
    """

    def __init__(self, dataset, batch_size, k=2):
        self.dataset = dataset
        self.k = k
        self.p = batch_size // k

        if batch_size % k != 0:
            raise ValueError(f"Batch size ({batch_size}) must be divisible by K ({k})")

        # Map authors to their data indices
        self.author_to_indices = defaultdict(list)
        for idx, item in enumerate(dataset):
            self.author_to_indices[item["labels"]].append(idx)

        self.authors = list(self.author_to_indices.keys())

    def __iter__(self):
        # Shuffle authors
        random.shuffle(self.authors)

        # Create a randomized copy of indices for popping
        author_to_indices_copy = {
            author: random.sample(indices, len(indices))
            for author, indices in self.author_to_indices.items()
        }

        batches = []
        # Keep sampling as long as we have enough authors with at least K texts left
        available_authors = [a for a in self.authors if len(author_to_indices_copy[a]) >= self.k]

        while len(available_authors) >= self.p:
            # Pick P random authors
            selected_authors = random.sample(available_authors, self.p)

            for author in selected_authors:
                # Extract exactly K texts per author
                for _ in range(self.k):
                    batches.append(author_to_indices_copy[author].pop())

            # Update available authors
            available_authors = [a for a in available_authors if len(author_to_indices_copy[a]) >= self.k]

        # Returns a flat list of indices. Hugging Face's BatchSampler will chunk this
        # into exactly batch_size, naturally resulting in P authors * K texts per batch!
        return iter(batches)

    def __len__(self):
        return len(self.dataset)


# ==========================================
#  CUSTOM TRAINER (P-K SAMPLER + BATCH-HARD)
# ==========================================
class AuthorTripletTrainer(Trainer):
    def __init__(self, triplet_margin=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.triplet_margin = triplet_margin

    def _get_train_sampler(self, *args, **kwargs):
        batch_size = self.args.train_batch_size
        return PKSampler(self.train_dataset, batch_size=batch_size, k=2)

    def _get_eval_sampler(self, eval_dataset, *args, **kwargs):
        batch_size = self.args.eval_batch_size
        return PKSampler(eval_dataset, batch_size=batch_size, k=2)

    # ---> NEW: Force the Trainer to calculate Triplet Loss during Evaluation <---
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Overrides evaluation step to ensure our custom Triplet Loss is calculated and returned."""
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
            # The Trainer expects a tuple of (loss, logits, labels)
            # We don't care about returning raw logits/labels for Triplet Loss evaluation, just the loss.
            return (loss, None, None)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. Extract true author IDs
        labels = inputs.pop("labels")

        # 2. Forward pass: Force model to output raw hidden states
        outputs = model(**inputs, output_hidden_states=True)

        # 3. Extract Hidden States for E5 or Llama
        if hasattr(outputs, "last_hidden_state"):
            token_embeddings = outputs.last_hidden_state
        else:
            token_embeddings = outputs.hidden_states[-1]

        # 4. Mean Pooling
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                        min=1e-9)

        # Normalize vectors to hypersphere
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # 5. SOTA BATCH-HARD TRIPLET LOSS
        # Calculate pairwise Euclidean distances
        dist_mat = torch.cdist(embeddings, embeddings, p=2)

        # Create Positive/Negative masks
        labels = labels.unsqueeze(1)
        is_pos = torch.eq(labels, labels.T).float()
        is_neg = 1.0 - is_pos

        # HARDEST POSITIVE (Max distance among same author)
        hardest_positive_dist = (dist_mat * is_pos).max(dim=1)[0]

        # HARDEST NEGATIVE (Min distance among different authors)
        # Add a huge penalty to same authors so they are never picked as the minimum
        max_dist = dist_mat.max().item()
        hardest_negative_dist = (dist_mat + (is_pos * (max_dist + 10.0))).min(dim=1)[0]
        #torch.nn.functional.normalize(embeddings, p=2
        #4 or 10 or 100 doesn't matter - just over 2 is perfect - so it will always take the negative one
        # Calculate Loss: max(0, hardest_pos - hardest_neg + margin)
        loss = torch.relu(hardest_positive_dist - hardest_negative_dist + self.triplet_margin)

        # Average the loss across the batch
        loss = loss.mean()

        # Fallback to keep gradient graph alive if batch is perfectly zeroed
        if loss.item() == 0.0:
            loss = loss + (embeddings.sum() * 0)

        return (loss, outputs) if return_outputs else loss