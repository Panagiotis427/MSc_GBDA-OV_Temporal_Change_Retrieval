"""
Change Feature Engineering Module.

This module implements the core change representation logic for temporal satellite imagery:
delta_f (difference vector) and concatenated representations. These features are then mapped
to a multimodal space via a lightweight projection head.
"""
import torch
from typing import Tuple, Union


def compute_change_feature(
    embedding_T1: torch.Tensor,
    embedding_T2: torch.Tensor,
    mode: str = "difference"
) -> torch.Tensor:
    """
    Compute the change feature representation from two temporal embeddings.

    Args:
        embedding_T1 (torch.Tensor): First time-step embedding. Shape: [batch_size, embed_dim].
        embedding_T2 (torch.Tensor): Second time-step embedding. Shape: [batch_size, embed_dim].
        mode (str): Feature computation mode.
            - "difference": delta_f = f_T2 - f_T1 (default)
            - "concatenate": [f_T1, f_T2] concatenated

    Returns:
        torch.Tensor: Change feature representation. Shape depends on mode:
            - difference: [batch_size, embed_dim]
            - concatenate: [batch_size, 2 * embed_dim]

    Example:
        >>> emb_t1 = torch.randn(4, 768)  # batch of 4, CLIP ViT-L/14 dim
        >>> emb_t2 = torch.randn(4, 768)
        >>> delta = compute_change_feature(emb_t1, emb_t2, mode="difference")
        >>> print(delta.shape)  # torch.Size([4, 768])
    """
    assert embedding_T1.size() == embedding_T2.size(), \
        f"Embeddings must have same shape: {embedding_T1.size()} vs {embedding_T2.size()}"

    if mode.lower() == "difference":
        delta_f = embedding_T2 - embedding_T1
    elif mode.lower() == "concatenate":
        delta_f = torch.cat([embedding_T1, embedding_T2], dim=-1)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'difference' or 'concatenate'.")

    return delta_f


def compute_change_magnitude(
    embedding_T1: torch.Tensor,
    embedding_T2: torch.Tensor
) -> torch.Tensor:
    """
    Compute the magnitude of change (L2 norm of difference vector).

    Useful for filtering out negligible changes or weighting by change intensity.

    Args:
        embedding_T1 (torch.Tensor): First time-step embedding. Shape: [batch_size, embed_dim].
        embedding_T2 (torch.Tensor): Second time-step embedding. Shape: [batch_size, embed_dim].

    Returns:
        torch.Tensor: L2 norm of delta_f for each sample. Shape: [batch_size].

    Example:
        >>> emb_t1 = torch.randn(4, 768)
        >>> emb_t2 = torch.randn(4, 768) + 0.5  # slight shift
        >>> magnitudes = compute_change_magnitude(emb_t1, emb_t2)
        >>> print(magnitudes.shape)  # torch.Size([4])
    """
    delta_f = embedding_T2 - embedding_T1
    return torch.norm(delta_f, p=2, dim=-1)


def normalize_embeddings(
    embeddings: torch.Tensor,
    method: str = "l2"
) -> torch.Tensor:
    """
    Normalize embeddings to unit length for cosine similarity computation.

    Args:
        embeddings (torch.Tensor): Input tensor. Shape: [..., embed_dim].
        method (str): Normalization method. Only 'l2' supported.

    Returns:
        torch.Tensor: Normalized embeddings with same shape as input.
    """
    if method.lower() == "l2":
        norm = torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        # Avoid division by zero: only add epsilon when norm is very small
        return embeddings / torch.where(norm > 1e-8, norm, torch.ones_like(norm) * 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
