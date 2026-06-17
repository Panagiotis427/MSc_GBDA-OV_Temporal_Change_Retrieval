"""
Model Architecture: Lightweight Projection Head.

This module implements the trainable parameters of the system: a multi-layer
perceptron (MLP) with dropout and LayerNorm that maps change features to a
multimodal space. The CLIP backbone is frozen — only this lightweight adapter is
learned. The contrastive loss that trains it lives with each trainer
(``train._masked_infonce`` for the PEFT ProjectionHead; ``lora_train._infonce_loss``
for LoRA).
"""
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple


class ProjectionHead(nn.Module):
    """
    Lightweight MLP adapter that maps change features to multimodal space.

    This is the only trainable component of the system. The frozen CLIP vision/backbone
    encoders are passed through a small neural network with dropout for regularization
    and LayerNorm for training stability.

    Architecture:
        Input (delta_f): [batch_size, embed_dim]  e.g., [B, 768]
                          - For difference mode: same dimension as CLIP embeddings
                          - For concatenation: double the dimension (e.g., 1536)
        Hidden layers: ReLU + Dropout + LayerNorm (repeated for capacity)
        Output: [batch_size, embed_dim]  Projected to match text embedding space

    Example:
        >>> model = ProjectionHead(input_dim=768, hidden_dim=512, output_dim=768)
        >>> delta_f = torch.randn(32, 768)  # batch of 32 change pairs
        >>> projected = model(delta_f)
        >>> print(projected.shape)  # torch.Size([32, 768])
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        hidden_dims: Tuple[int, ...] = (512, 256),
        dropout_rate: float = 0.3,
        layer_norm_eps: float = 1e-6
    ):
        """
        Initialize the projection head.

        Args:
            input_dim (int): Dimensionality of change features (delta_f).
                Typically matches CLIP embedding dimension (768 for ViT-L/14).
            output_dim (int, optional): Output dimension. Defaults to input_dim.
            hidden_dims (tuple): Intermediate layer dimensions. More layers = more
time but better capacity.
            dropout_rate (float): Dropout probability after each ReLU. 0.3 is a good
                starting point for regularization without underfitting.
            layer_norm_eps: Epsilon for numerical stability in LayerNorm.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.hidden_dims = hidden_dims

        # Build the MLP with alternating ReLU, Dropout, LayerNorm layers
        layers = []
        current_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear projection + ReLU activation
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            # LayerNorm after each non-linearity (stabilizes training)
            layers.append(nn.LayerNorm(hidden_dim, eps=layer_norm_eps))
            # Dropout for regularization
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            current_dim = hidden_dim

        # Final projection to output_dim with LayerNorm
        layers.append(nn.Linear(current_dim, self.output_dim))
        layers.append(nn.LayerNorm(self.output_dim, eps=layer_norm_eps))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights with Xavier/Glorot initialization (better for ReLU)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the projection head.

        Args:
            x (torch.Tensor): Input change features. Shape: [batch_size, input_dim].

        Returns:
            torch.Tensor: Projected embeddings in multimodal space.
                Shape: [batch_size, output_dim]
        """
        return self.mlp(x)

    def num_parameters(self) -> int:
        """
        Count trainable parameters in the projection head.

        Returns:
            int: Total number of trainable parameters. Should be << CLIP backbone size.
                For input_dim=768, hidden_dims=(512,256), output_dim=768:
                ~0.5M params vs 350M+ in full CLIP ViT-L/14
        """
        return sum(p.numel() for p in self.parameters())

    def num_floating_point_ops(self, input_shape: Tuple[int]) -> int:
        """
        Estimate floating point operations per forward pass.

        Args:
            input_shape (tuple): Shape of input tensor as (batch_size, input_dim).

        Returns:
            int: Estimated FLOPs for the entire MLP.
        """
        batch_size, in_dim = input_shape
        total_flops = 0

        current_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            # Linear layer: 2 * in_features * out_features (weights + bias)
            flops = 2 * current_dim * hidden_dim * batch_size
            total_flops += flops
            current_dim = hidden_dim

        # Final linear projection
        flops = 2 * current_dim * self.output_dim * batch_size
        total_flops += flops

        return int(total_flops)


def create_projection_head(
    input_dim: int = 768,
    output_dim: Optional[int] = None,
    hidden_dims: Tuple[int, ...] = (512, 256),
    dropout_rate: float = 0.3
) -> ProjectionHead:
    """
    Factory function for creating projection heads with sensible defaults.

    Args:
        input_dim: Dimensionality of change features (CLIP embedding size).
            Default 768 matches CLIP ViT-L/14.
        output_dim: Output dimension. Defaults to input_dim.
        hidden_dims: MLP bottleneck layers. Larger = more capacity, slower training.
        dropout_rate: Regularization strength. 0.3 is a good starting point.

    Returns:
        ProjectionHead: Configured adapter network.

    Example usage for different scenarios:
        # Default (matches CLIP ViT-L/14)
        >>> head = create_projection_head()
        >>> print(head.num_parameters())  # ~500k params

        # More capacity for hard negative mining
        >>> head = create_projection_head(
        ...     hidden_dims=(1024, 512, 256),
        ...     dropout_rate=0.2
        ... )

        # Minimal adapter (faster training, less expressive)
        >>> head = create_projection_head(
        ...     hidden_dims=(256,),
        ...     output_dim=768
        ... )
    """
    return ProjectionHead(input_dim, output_dim, hidden_dims, dropout_rate)


def save_adapter(path: str, adapter: "ProjectionHead", meta: Dict[str, Any]) -> None:
    """Persist a trained adapter together with the metadata needed to rebuild
    an identical ``ProjectionHead`` (dims, feature mode, encoder it targets)."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": adapter.state_dict(), "meta": meta}, path)


def load_adapter(path: str, map_location="cpu") -> Tuple["ProjectionHead", Dict[str, Any]]:
    """Reconstruct a ``ProjectionHead`` from a checkpoint written by
    :func:`save_adapter`. Returns ``(adapter_in_eval_mode, meta)``."""
    # weights_only=True: the adapter path is derived from user CLI dataset/encoder
    # strings, and save_adapter only ever stores tensors + a primitive meta dict
    # (dims/feature_mode/names) — so the safe loader suffices and we refuse to
    # unpickle arbitrary objects from a .pt that happens to sit at that path.
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    meta = ckpt["meta"]
    adapter = ProjectionHead(
        input_dim=meta["input_dim"],
        output_dim=meta["output_dim"],
        hidden_dims=tuple(meta["hidden_dims"]),
        dropout_rate=meta.get("dropout_rate", 0.3),
    )
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()
    return adapter, meta
