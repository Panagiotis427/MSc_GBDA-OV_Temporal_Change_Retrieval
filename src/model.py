"""
Model Architecture: Lightweight Projection Head and InfoNCE Loss.

This module implements the trainable parameters of the system:
a multi-layer perceptron (MLP) with dropout and LayerNorm that maps change features
to a multimodal space, and contrastive loss for training.
The CLIP backbone is frozen - only these lightweight adapters are learned.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


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


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) Loss for contrastive learning.

    This loss maximizes cosine similarity between matched (positive) pairs
    while minimizing similarity with all other pairs treated as negatives.

    The formula:
        L = - log(exp(sim(anchor, positive) / τ) / Σ_j exp(sim(anchor, j) / τ))

    where:
        - anchor: change feature from one temporal pair (delta_f)
        - positive: text embedding of the true description
        - negatives: all other embeddings in batch (text or image pairs)
        - τ (temperature): controls sharpness. Lower = sharper, higher = smoother.

    Example:
        >>> # Training with matched pairs
        >>> anchor = torch.randn(8, 768)      # 4 change features x 2 temporal views
        >>> positive = torch.randn(4, 768)    # True text descriptions
        >>> negative_text = torch.randn(4, 768)  # Wrong text descriptions
        >>>
        >>> criterion = InfoNCELoss()
        >>> loss = criterion(anchor, positive, negative=negative_text)

    Example with temperature scaling:
        >>> # Standard: τ=0.1 (sharp distribution, good for hard negatives)
        >>> criterion = InfoNCELoss(temperature=0.1)
        >>>
        >>> # Smooth: τ=1.0 (easier optimization, softer gradients)
        >>> criterion = InfoNCELoss(temperature=1.0)
    """

    def __init__(self, temperature: float = 0.1):
        """
        Initialize the InfoNCE loss.

        Args:
            temperature (float): Temperature parameter τ. Lower values (<1) emphasize
                high-similarity pairs and make learning harder but more discriminative.
                Higher values (>1) flatten the distribution for easier training.
                Typical range: [0.05, 1.0]. Default 0.1 works well for CLIP-style tasks.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        batch_size: int = None,
        device: Optional[torch.device] = None,
        temperature: Optional[float] = None
    ) -> Tuple[float, torch.Tensor]:
        """
        Compute InfoNCE loss for contrastive pairs.

        Args:
            anchor (torch.Tensor): Anchor embeddings. Shape: [2 * batch_size, embed_dim]
                (using 2 views per sample improves gradient signal).
            positive (torch.Tensor): Positive text embeddings matched to anchors.
                Shape: [batch_size, embed_dim].
            negative (torch.Tensor, optional): Negative text/image embeddings for
                stronger contrastive signal. If None, only uses positives as reference.
            batch_size (int, optional): Explicit batch size. Auto-detected if not provided.
            device (torch.device, optional): Device to move tensors on.

        Returns:
            tuple: (loss_value, logits) where:
                - loss_value (float): Computed InfoNCE loss scalar.
                - logits (torch.Tensor): Full attention matrix before softmax. Shape:
                  [2*batch_size, num_positives + num_negatives] for debugging.

        Raises:
            ValueError: If input dimensions don't match or if tensors are empty.
        """
        # Auto-detect batch size
        if batch_size is None:
            batch_size = anchor.shape[0] // 2

        assert positive.shape[0] == batch_size and len(positive.shape) == 2, \
            f"Positive embeddings must be [batch_size, embed_dim], got {positive.shape}"

        # Use provided temperature or default
        tau = temperature if temperature is not None else self.temperature

        # Compute cosine similarities with temperature scaling
        # Similarity: (a · p) / τ = cos(θ) / τ
        sim_matrix = self._compute_similarities(
            anchor.to(device), positive.to(device), temperature=tau
        )  # Shape: [2*batch_size, batch_size] - each row is anchor i's similarity to all positives

        if negative is not None:
            neg_batch_size = negative.shape[0]
            assert neg_batch_size == batch_size, \
                f"Negative batch {neg_batch_size} != positive batch {batch_size}"

            # Compute similarities: each anchor vs all negatives in the batch
            neg_sim_matrix = self._compute_similarities(
                anchor.to(device), negative.to(device)
            )  # Shape: [2*batch_size, batch_size]

            # Build logits matrix: [num_anchors, num_negatives + num_positives]
            total_logits = torch.zeros(2 * batch_size, 2 * batch_size,
                                      device=sim_matrix.device, dtype=sim_matrix.dtype)
            total_logits[:, :batch_size] = neg_sim_matrix  # First half are negatives
            total_logits[:, batch_size:2*batch_size] = sim_matrix  # Second half are positives
        else:
            # Only positives - build logits matrix where each anchor i compares to all positives [0..batch_size-1]
            # Shape: [2*batch_size, batch_size], row i contains similarities of anchor i to all batch positives
            total_logits = sim_matrix.clone()  # Already shape [2*batch_size, batch_size]

        # Apply softmax and compute InfoNCE loss
        log_probs = nn.functional.log_softmax(total_logits, dim=-1)

        # Matched pairs: anchor i matches positive i for each i in [0..batch_size-1]
        # Since we have 2*batch_size anchors but only batch_size positives,
        # we take the first pair from each (anchor 0->pos 0, anchor 1->pos 1, etc.)
        matched_indices = torch.arange(batch_size)  # [0, 1, 2, ..., batch_size-1]
        loss = -torch.mean(log_probs[matched_indices, matched_indices])

        return loss, total_logits

    def _compute_similarities(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute cosine similarity matrix between two sets of embeddings.

        Args:
            x (torch.Tensor): First set. Shape: [n, embed_dim].
            y (torch.Tensor): Second set. Shape: [m, embed_dim].
            temperature (float, optional): Temperature scaling parameter τ.
                Defaults to self.temperature if not provided.

        Returns:
            torch.Tensor: Cosine similarity matrix with temperature scaling. Shape: [n, m].
                Each entry: cos(θ) / τ = x_i · y_j / (||x_i|| * ||y_j|| * τ)
        """
        # Use provided temperature or default
        tau = temperature if temperature is not None else self.temperature

        # L2 normalize both sets
        norm_x = nn.functional.normalize(x, dim=-1)
        norm_y = nn.functional.normalize(y, dim=-1)

        # Matrix multiplication gives cosine similarities with temperature scaling
        return torch.matmul(norm_x, norm_y.T) / tau

    def forward_with_hard_negatives(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        all_positives: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None
    ) -> Tuple[float, dict]:
        """
        InfoNCE with hard negative mining (advanced usage).

        This variant identifies the most confusing negatives (highest similarity to anchor)
        and uses only those for the loss. Harder negatives → better discrimination.

        Args:
            anchor: Change features [2*batch_size, dim]
            positive: Matched text embeddings [batch_size, dim]
            all_positives: All validation positives for mining hard negatives.
                If None, uses only current batch as negatives. Shape: [num_all_pos, dim].
            temperature: Override the default temperature parameter.

        Returns:
            tuple: (loss, metrics) where metrics dict includes:
                - 'hard_negative_count': Number of samples with hard negatives used
                - 'sim_at_hard_neg': Average similarity at hard negative threshold
        """
        if all_positives is None or len(all_positives) == 0:
            return self.forward(anchor, positive, batch_size=anchor.shape[0] // 2)

        tau = temperature or self.temperature
        n_anchors = anchor.shape[0]
        n_all_pos = len(all_positives)

        # Compute similarity to ALL positives (including current batch)
        sim_matrix = torch.matmul(
            nn.functional.normalize(anchor, dim=-1),
            nn.functional.normalize(all_positives, dim=-1).T / tau
        )  # [2*batch_size, num_all_pos]

        # Identify hard negatives: samples where wrong positives have high similarity
        threshold = torch.topk(sim_matrix.max(dim=1)[0], k=5, largest=True)[0] - 0.3
        is_hard_neg = sim_matrix > threshold.unsqueeze(-1)

        if is_hard_neg.sum() == 0:
            # No hard negatives found, fall back to standard InfoNCE
            return self.forward(anchor, positive)

        # Build loss from only hard negative pairs
        num_hard = is_hard_neg.sum()
        hard_loss = -torch.mean(
            torch.log(torch.softmax(sim_matrix[is_hard_neg], dim=-1) + 1e-8)
        )

        return hard_loss, {
            "hard_negative_count": int(num_hard),
            "sim_at_threshold": threshold.cpu().numpy()
        }
