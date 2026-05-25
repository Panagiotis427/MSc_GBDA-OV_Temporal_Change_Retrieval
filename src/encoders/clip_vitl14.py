"""
OpenAI CLIP ViT-L/14 wrapped as an `ImageTextEncoder`.

Composes the existing `FrozenTextEncoder` for text and loads a `CLIPModel` /
`CLIPProcessor` pair for the image side. Exposes `_clip_model` and `_processor`
attributes so the legacy `load_parquet_as_embeddings` function in
`src/data_loader.py` can keep using this encoder unchanged.
"""
from __future__ import annotations

import os
from src import _cache  # noqa: F401  sets HF_HOME before transformers
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ..text_encoder import FrozenTextEncoder


_DEFAULT_MODEL = "openai/clip-vit-large-patch14"


class CLIPViTL14Encoder:
    """`ImageTextEncoder` for OpenAI CLIP ViT-L/14 (768-d shared space)."""

    name = "clip_vitl14"
    embed_dim = 768
    image_input_size = 224

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[torch.device] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from src import _cache
        self.cache_dir = cache_dir or _cache.CLIP_CACHE_DIR

        self._text = FrozenTextEncoder(
            model_name=model_name,
            device=self.device,
            cache_dir=cache_dir,
        )

        print(f"Loading CLIP vision tower: {model_name}")
        self._clip_model: CLIPModel = CLIPModel.from_pretrained(model_name, cache_dir=self.cache_dir).to(self.device)
        self._clip_model.eval()
        for p in self._clip_model.parameters():
            p.requires_grad = False
        self._processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_name, cache_dir=self.cache_dir)

    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            t = self._text.encode(batch)
            t = F.normalize(t, dim=-1)
            embs.append(t.detach().cpu().numpy())
        return np.concatenate(embs, axis=0)

    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
        batch_size: int = 32,
    ) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]
        embs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                pixel_values = self._processor(images=batch, return_tensors="pt").pixel_values.to(self.device)
                vision_out = self._clip_model.vision_model(pixel_values=pixel_values)
                feats = self._clip_model.visual_projection(vision_out.pooler_output)
                feats = F.normalize(feats, dim=-1)
                embs.append(feats.cpu().numpy())
        return np.concatenate(embs, axis=0)

    def compute_patch_text_similarity(
        self,
        image: Image.Image,
        text: str,
    ) -> np.ndarray:
        """Per-patch cosine similarity, projected into the shared CLIP space.

        Returns a ``[grid_h, grid_w]`` float32 array in ``[0, 1]`` derived from
        per-patch · text-embedding cosine similarity. For ViT-L/14 at 224 input
        the grid is 16×16.
        """
        with torch.no_grad():
            pixel_values = self._processor(images=image, return_tensors="pt").pixel_values.to(self.device)
            vision_outputs = self._clip_model.vision_model(pixel_values=pixel_values)
            patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
            projected = self._clip_model.visual_projection(patch_tokens)
            projected = F.normalize(projected, dim=-1)

            text_inputs = self._processor.tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            text_features = self._clip_model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=-1)

            sims = (projected @ text_features.t()).squeeze(0).squeeze(-1)
            n_patches = sims.shape[0]
            grid_side = int(round(n_patches**0.5))
            if grid_side * grid_side != n_patches:
                raise RuntimeError(
                    f"Expected square patch grid; got {n_patches} patches (sqrt={grid_side})."
                )
            grid = sims.view(grid_side, grid_side).cpu().numpy().astype(np.float32)

        lo, hi = float(grid.min()), float(grid.max())
        if hi - lo < 1e-8:
            return np.zeros_like(grid, dtype=np.float32)
        return ((grid - lo) / (hi - lo)).astype(np.float32)
