"""
Change retrieval over a pair corpus.

Given a :class:`~src.embeddings.PairEmbeddingStore` (cached ``f_T1, f_T2`` for
every pair) and a text encoder, rank all pairs by how well the bi-temporal
*change* matches a natural-language query. Three scoring approaches:

- ``naive``      : cos(text, f_T2)              — image retrieval, no change
                   modelling (lower-bound baseline).
- ``zero_shot``  : cos(text, f_T2) - cos(text, f_T1)
                   — Δ-similarity; pure zero-shot, captures *directional*
                   change. Equivalent to text · (f_T2 - f_T1).
- ``peft``       : cos(text, g(Δf)) where ``g`` is a trained ``ProjectionHead``
                   adapter mapping the change feature into text space.

The corpus for a DEN subset is small (tens–hundreds of pairs), so scoring is
an exact dense matmul — no ANN index needed (and Δ-similarity is not a single
inner-product NN query anyway).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.datasets.base import PairKey
from src.embeddings import PairEmbeddingStore

NAIVE = "naive"
ZERO_SHOT = "zero_shot"
PEFT = "peft"
APPROACHES = (NAIVE, ZERO_SHOT, PEFT)


@dataclass
class RetrievalResult:
    pair: PairKey
    score: float
    rank: int


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


class ChangeRetriever:
    def __init__(
        self,
        store: PairEmbeddingStore,
        encoder,
        adapter: Optional[torch.nn.Module] = None,
        feature_mode: str = "difference",
    ) -> None:
        if encoder.embed_dim != store.embed_dim:
            raise ValueError(
                f"Encoder dim {encoder.embed_dim} != store dim {store.embed_dim}. "
                "Embeddings must be recomputed with this encoder."
            )
        self.store = store
        self.encoder = encoder
        self.adapter = adapter
        self.feature_mode = feature_mode
        self._f_t1 = _l2(store.f_t1)
        self._f_t2 = _l2(store.f_t2)
        self._peft_proj: Optional[np.ndarray] = None  # cached g(Δf), L2-normed

    # ------------------------------------------------------------------
    def set_adapter(self, adapter: Optional[torch.nn.Module],
                     feature_mode: Optional[str] = None) -> None:
        self.adapter = adapter
        if feature_mode is not None:
            self.feature_mode = feature_mode
        self._peft_proj = None  # invalidate cache

    def _project_changes(self) -> np.ndarray:
        if self.adapter is None:
            raise RuntimeError("PEFT approach requested but no adapter is set.")
        if self._peft_proj is None:
            delta = self.store.change_features(mode=self.feature_mode)
            self.adapter.eval()
            dev = next(self.adapter.parameters()).device
            with torch.no_grad():
                g = self.adapter(torch.from_numpy(delta).float().to(dev))
            self._peft_proj = _l2(g.cpu().numpy().astype(np.float32))
        return self._peft_proj

    # ------------------------------------------------------------------
    def score_all(self, query: str, approach: str = ZERO_SHOT) -> np.ndarray:
        """Return a score per pair, aligned with ``store.pairs`` (higher = better)."""
        t = self.encoder.encode_text(query)[0].astype(np.float32)  # (D,), L2-normed
        if approach == NAIVE:
            return self._f_t2 @ t
        if approach == ZERO_SHOT:
            return self._f_t2 @ t - self._f_t1 @ t
        if approach == PEFT:
            return self._project_changes() @ t
        raise ValueError(f"Unknown approach {approach!r}; use one of {APPROACHES}")

    def search(
        self,
        query: str,
        approach: str = ZERO_SHOT,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        scores = self.score_all(query, approach)
        order = np.argsort(-scores)[:top_k]
        return [
            RetrievalResult(self.store.pairs[i], float(scores[i]), r)
            for r, i in enumerate(order)
        ]
