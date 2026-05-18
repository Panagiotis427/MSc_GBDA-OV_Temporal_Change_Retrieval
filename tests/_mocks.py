"""
Shared deterministic mock encoder for the fast test suite (no CLIP, no
network). Tests subclass :class:`MockEncoderBase` and override
``encode_text`` with their own destination-class / caption-aware mapping.

The class/palette constants mirror ``scripts.make_den_fixture._PALETTE`` so
``encode_image`` recovers a tile's dominant class via per-pixel
nearest-palette + majority vote — same dominant class the labels report.
"""
from __future__ import annotations

import numpy as np
import torch

from src.datasets._palette import DEN_CLASS_NAMES, DEN_PALETTE

_CLASSES = list(DEN_CLASS_NAMES)
_PALETTE = DEN_PALETTE


class MockEncoderBase:
    """Implements ``encode_image`` (palette-majority one-hot) +
    ``compute_patch_text_similarity`` (deterministic gradient). Subclasses
    must provide ``encode_text``."""

    name = "mock"
    embed_dim = len(_CLASSES)
    image_input_size = 8
    device = torch.device("cpu")

    def _onehot(self, cls: str) -> np.ndarray:
        v = np.zeros(self.embed_dim, dtype=np.float32)
        v[_CLASSES.index(cls)] = 1.0
        return v

    def encode_image(self, images, batch_size: int = 32) -> np.ndarray:
        if not isinstance(images, list):
            images = [images]
        pal = np.array([_PALETTE[c] for c in _CLASSES], dtype=np.float32)
        out = []
        for im in images:
            px = np.array(im.convert("RGB"), dtype=np.float32).reshape(-1, 3)
            nearest = np.argmin(((px[:, None, :] - pal[None]) ** 2).sum(-1), axis=1)
            idx = int(np.bincount(nearest, minlength=self.embed_dim).argmax())
            out.append(self._onehot(_CLASSES[idx]))
        return np.stack(out)

    def encode_text(self, texts, batch_size: int = 32) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError("subclass must provide encode_text")

    def compute_patch_text_similarity(self, image, text) -> np.ndarray:
        return np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
