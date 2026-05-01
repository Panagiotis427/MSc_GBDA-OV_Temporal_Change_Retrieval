"""
Training Pipeline: Projection Head with InfoNCE Contrastive Learning.

This module implements the complete training loop for Week 4:
- Optimizer setup and gradient-based learning
- MLflow integration for metrics tracking (loss, LR, epochs)
- Evaluation using Recall@K and Mean Average Precision (mAP)
"""
import os
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# MLflow integration for experiment tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    print("WARNING: MLflow not installed. Metrics will not be logged.")
    MLFLOW_AVAILABLE = False

from src.model import ProjectionHead, InfoNCELoss, create_projection_head
from src.text_encoder import FrozenTextEncoder
import time

# =============================================================================
# Task 4.1: Training Script - Standard PyTorch training loop
# =============================================================================

class TemporalPairDataset(Dataset):
    """
    Simple dataset for (change_features, anchor_texts, positive_text) tuples.
    For InfoNCE contrastive learning with projected change features and text embeddings.
    """

    def __init__(self,
                 change_features: np.ndarray,
                 anchor_texts: List[str],
                 positive_texts: List[str],
                 text_encoder: FrozenTextEncoder):
        # Validate inputs are not empty
        if len(anchor_texts) == 0:
            raise ValueError("anchor_texts cannot be empty")
        if len(positive_texts) == 0:
            raise ValueError("positive_texts cannot be empty")
        if change_features.shape[0] != len(anchor_texts):
            raise ValueError(
                f"change_features has {change_features.shape[0]} samples but "
                f"anchor_texts has {len(anchor_texts)} entries"
            )
        if change_features.shape[0] != len(positive_texts):
            raise ValueError(
                f"change_features has {change_features.shape[0]} samples but "
                f"positive_texts has {len(positive_texts)} entries"
            )

        self.change_features = torch.from_numpy(change_features).float()
        self.anchor_texts = anchor_texts
        self.positive_texts = positive_texts
        self.text_encoder = text_encoder

    def __len__(self) -> int:
        return len(self.anchor_texts)

    def __getitem__(self, idx: int):
        with torch.no_grad():
            change_emb = self.change_features[idx]
            # Normalize the change feature using CLIP's vision tower normalization
            projected_change = nn.functional.normalize(change_emb, dim=-1)
        anchor_text = self.anchor_texts[idx]
        positive_text = self.positive_texts[idx]
        return change_emb, anchor_text, positive_text


def train_model(adapter: ProjectionHead, dataset: TemporalPairDataset,
                text_encoder: FrozenTextEncoder,
                batch_size: int = 32, num_epochs: int = 10, learning_rate: float = 1e-4,
                device: Optional[torch.device] = None, val_dataset: Optional[TemporalPairDataset] = None,
                eval_freq: int = 1) -> Dict:
    """
    Train the Projection Head using InfoNCE contrastive learning.

    Args:
        adapter: The trainable MLP projection head (~0.5M params)
        dataset: Training data with projected change features, anchors, positives
        text_encoder: Frozen CLIP text encoder for encoding anchor/positive texts
        batch_size: Batch size for DataLoader
        num_epochs: Number of training epochs
        learning_rate: Adam optimizer learning rate (default 1e-4 works well)
        device: 'cuda' or 'cpu'
        val_dataset: Optional validation set for periodic evaluation
        eval_freq: How many epochs to wait between evaluations

    Returns:
        dict with training history and final metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adapter.to(device)
    print(f"Training on {device}")

    # Optimizer: Adam with default betas works well for small adapters
    optimizer = torch.optim.Adam(adapter.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = InfoNCELoss(temperature=0.1).to(device)
    pin_memory = torch.device(device).type == 'cuda' if isinstance(device, str) else device.type == 'cuda'
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)

    history = {'train_loss': [], 'val_recall_at_k': {}, 'learning_rate': []}

    for epoch in range(num_epochs):
        adapter.train()
        total_loss = 0.0
        num_batches = 0

        # Collect all anchor and positive texts first (needed for batched text encoding)
        all_texts = [[], []]  # [anchor_texts, positive_texts]
        for change_emb, anchor_text, positive_text in train_loader:
            all_texts[0].append(anchor_text)
            all_texts[1].append(positive_text)

        # Encode all target text embeddings at once (the "positives")
        with torch.no_grad():
            target_embeddings = text_encoder.encode(all_texts[1]).to(device)  # [B, 768]

        # Process in batches to avoid OOM and for efficiency
        for i in range(0, len(target_embeddings), batch_size):
            B = min(batch_size, len(target_embeddings) - i)
            change_batch = torch.stack([dataset[i + j][0] for j in range(B)]).to(device)
            target_emb = target_embeddings[i:i+B]

            projected_change = adapter(change_batch)  # [B, 768]

            # Compute cosine similarity: maximize mean(similarity of matched pairs)
            norm_proj = nn.functional.normalize(projected_change, dim=-1)
            norm_target = nn.functional.normalize(target_emb, dim=-1)
            sim_matrix = torch.matmul(norm_proj, norm_target.T)  # [B, B]

            tau = criterion.temperature
            logits = sim_matrix / tau

            matched_indices = torch.arange(B)
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            loss = -log_probs[matched_indices, matched_indices].mean()

            # Backpropagate gradients and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches
        history['train_loss'].append(avg_train_loss)
        history['learning_rate'].append(scheduler.get_last_lr()[0])

        if val_dataset is not None and (epoch + 1) % eval_freq == 0:
            recall_at_k, mAP = evaluate_retrieval(adapter, val_dataset, device=device, text_encoder=text_encoder)
            history['val_recall_at_k'][f"recall@{eval_freq}"] = recall_at_k.get(f"recall@{eval_freq}", "N/A")
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_train_loss:.4f} | Recall@5: {recall_at_k.get('recall@5', 'N/A'):.3f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_train_loss:.4f}")

        if MLFLOW_AVAILABLE and val_dataset is not None:
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

        scheduler.step()
    print(f"\nFinal loss: {history['train_loss'][-1]:.4f}")
    return history


def evaluate_retrieval(adapter, dataset, device, text_encoder, top_k=[1, 5, 10, 20, 50]):
    """
    Evaluate retrieval performance using Recall@K and Mean Average Precision.

    This function computes:
    - **Recall@K**: For each query, is the true positive within top-K results?
    - **Mean Average Precision (mAP)**: Average precision across all queries, then averaged over query pairs

    Args:
        adapter: Trained projection head
        dataset: Dataset with change features and text pairs
        device: Device to run evaluation on
        top_k: List of K values for Recall@K computation

    Returns:
        tuple: (recall_dict, mAP)
            - recall_dict: Dictionary mapping k -> recall@k score
            - mAP: Mean Average Precision across all queries
    """
    adapter.eval()
    all_changes = []  # All change feature projections [N, 768]
    all_queries = []   # Query embeddings for each sample [N, 768]
    all_positives = [] # Positive match indices (self-match in this dataset) [N, 768]

    with torch.no_grad():
        for change_emb, anchor_text, positive_text in dataset:
            change_batch = change_emb.unsqueeze(0).to(device)  # [1, 768]
            query_emb = text_encoder.encode(anchor_text).to(device)
            pos_emb = text_encoder.encode(positive_text).to(device)
            all_changes.append(adapter(change_batch))
            all_queries.append(query_emb)
            all_positives.append(pos_emb)

    N = len(dataset)
    all_changes = torch.cat(all_changes, dim=0)  # [N, 768]
    all_queries = torch.cat(all_queries, dim=0)  # [N, 768]
    all_positives = torch.cat(all_positives, dim=0)  # [N, 768]

    recall_dict = {}
    ap_scores = []

    for i in range(N):
        query_emb = all_queries[i].unsqueeze(0).to(device)
        pos_emb = all_positives[i].unsqueeze(0).to(device)

        with torch.no_grad():
            # Cosine similarities: query vs all changes and positive vs all changes
            # Shape: [1, 2N] - first N are query similarities, next N are positive similarities
            query_sims = torch.matmul(query_emb, all_changes.t())  # [1, N]
            pos_sims = torch.matmul(pos_emb, all_changes.t())  # [1, N]
            sims = torch.cat([query_sims, pos_sims], dim=-1)  # [1, 2N]
            ranked_idx = torch.argsort(sims.squeeze(0), descending=True).cpu().numpy()

        # Track true positives at each rank position
        true_pos_rank = [False] * (2 * N)

        # The positive match for sample i is at index (i + N) in the combined list
        # Because first N are query similarities, next N are positive similarities
        true_positive_idx = i + N
        true_pos_rank[ranked_idx[true_positive_idx]] = True

        # Compute Average Precision for this query
        num_relevant = 1  # Only one positive per sample (self-match)
        ap_hits = 0
        hits_at_k = {}

        for k in top_k:
            if k > len(ranked_idx):
                continue  # Skip k values larger than ranked_idx size
            if ranked_idx[k-1] == true_positive_idx:  # Target is within top-k
                hits_at_k[f"recall@{k}"] = True
            else:
                hits_at_k[f"recall@{k}"] = False

        for rank in range(2 * N):
            idx = ranked_idx[rank]
            if true_pos_rank[idx]:  # Found the positive match
                break
            elif true_pos_rank[idx]:
                ap_hits += 1

        avg_precision = ap_hits / num_relevant if num_relevant > 0 else 0.0
        ap_scores.append(avg_precision)

    # Compute recall@K for each k
    for k in top_k:
        if k > len(ranked_idx):
            continue  # Skip k values larger than ranked_idx size
        hit_count = sum(1 for i in range(N)
                       if ranked_idx[k-1] == (i + N))
        recall_dict[f"recall@{k}"] = hit_count / N if N > 0 else 0.0

    # Compute mean Average Precision across all queries
    mAP = np.mean(ap_scores)

    print(f"Retrieval Performance (N={N}):")
    for k, v in recall_dict.items():
        print(f"  {k}: {v:.3f}")
    print(f"  Mean Average Precision: {mAP:.4f}")

    return recall_dict, mAP


def main():
    """
    Main entry point for training the Projection Head.
    Usage: python -m src.train --epochs 10 --batch-size 32 --lr 1e-4
    """
    import argparse
    parser = argparse.ArgumentParser(description="Train Projection Head for temporal change retrieval")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )
    print(f"Device: {device}\n")

    # Load frozen text encoder
    print("Loading CLIP text encoder...")
    text_encoder = FrozenTextEncoder(cache_dir="/tmp/clip-training-cache")

    # Create synthetic dataset for demo (in production: use load_qfabric_features())
    torch.manual_seed(42)
    n_samples = 100
    change_features = np.random.randn(n_samples, 768).astype(np.float32) * 0.1
    anchor_texts = [f"industrial development {i}" for i in range(n_samples)]
    positive_texts = [f"construction at location {i}" for i in range(n_samples)]

    print(f"Creating dataset with {n_samples} samples...")
    train_dataset = TemporalPairDataset(change_features, anchor_texts, positive_texts, text_encoder)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize Projection Head
    adapter = create_projection_head(input_dim=768, hidden_dims=(512, 256), dropout_rate=0.3)
    print(f"Model params: {adapter.num_parameters():,}\n")

    # Train
    history = train_model(adapter, train_dataset, text_encoder,
                         batch_size=args.batch_size, num_epochs=args.epochs,
                         learning_rate=args.lr, device=device)

    # Save model
    save_path = "models/projection_head.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(adapter.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
