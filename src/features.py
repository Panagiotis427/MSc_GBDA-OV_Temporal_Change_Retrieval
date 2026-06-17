"""
Change Feature Engineering Module.

This module implements the core change representation logic for temporal satellite imagery:
delta_f (difference vector) and concatenated representations. These features are then mapped
to a multimodal space via a lightweight projection head.
"""
import torch


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
