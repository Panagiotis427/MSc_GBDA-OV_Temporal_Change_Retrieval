"""
Encoder-agnostic core.

Defines the `ImageTextEncoder` Protocol that decouples the rest of the pipeline
(retrieval, heatmaps, error analysis) from any specific vision-language model.

Two concrete implementations:
  - `src.encoders.clip_vitl14.CLIPViTL14Encoder` — OpenAI CLIP ViT-L/14 (768-d)
  - `src.encoders.georsclip.GeoRSCLIPEncoder` — RS-specific GeoRSCLIP (512-d) [session 2]
"""
from __future__ import annotations

from typing import List, Protocol, Union, runtime_checkable

import numpy as np
import torch
from PIL import Image


@runtime_checkable
class ImageTextEncoder(Protocol):
    """Protocol implemented by every encoder used in the project.

    Attributes:
        name: Short identifier (``"clip_vitl14"``, ``"georsclip"``).
        embed_dim: Output dimensionality of the shared multimodal space.
        image_input_size: Square side length the image preprocessor expects.
        device: Torch device the underlying model lives on.
    """

    name: str
    embed_dim: int
    image_input_size: int
    device: torch.device

    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return L2-normalised CPU ndarray of shape ``(N, embed_dim)``."""
        ...

    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return L2-normalised CPU ndarray of shape ``(N, embed_dim)``."""
        ...

    def compute_patch_text_similarity(
        self,
        image: Image.Image,
        text: str,
    ) -> np.ndarray:
        """Per-patch cosine similarity between image patches and the text query.

        Returns a 2-D ``np.float32`` array shaped ``(grid_h, grid_w)`` with
        values in ``[0, 1]``. Used by `src.heatmap.generate_heatmap`.
        """
        ...
