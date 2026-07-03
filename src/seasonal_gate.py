"""Image-level seasonal-robustness gate — stable-pair Δ-similarity false-positive rate.

A direct probe of the ``zero_shot`` approach's seasonal robustness, complementing
``seasonal_drift@K`` (the retrieval-side drift metric, which is uninformative on the
current corpora — no snow/ice positives — and is reported N/A in the benchmark).

The gate lifts the retrieval engine's ``zero_shot`` score to a whole-image binary
decision: for a change-description query ``t`` and a pair's L2-normalised global
embeddings ``f_T1, f_T2``,

    Δ(t) = cos(t, f_T2) − cos(t, f_T1)

is the *Δ-similarity* — how much more the query matches T2 than T1. For a **stable**
pair (no semantic change) ``f_T1 ≈ f_T2``, so Δ ≈ 0 for *any* query: a well-behaved
zero-shot scorer should not fire on stable pairs. The gate fires (predicts "change")
when ``Δ > threshold``; on stable pairs every firing is a **false positive**, so the
false-positive rate (FPR) over the stable subset, swept across thresholds, measures
exactly how often seasonal/illumination drift is mistaken for change.

Stable pairs are taken from the dataset's own ``PairLabel.stable`` flag
(``src.datasets.base.PairLabel``); for DEN that is ``total_change < stable_threshold``.
No new label logic is introduced here.

Query vectors are passed in already-encoded (shape ``(Q, D)``, L2-normalised — the
``ImageTextEncoder`` contract), decoupling text encoding (and any prompt ensembling)
from scoring; with multiple queries the per-pair score is the **max** Δ across queries
(the strictest "did *anything* falsely fire" reading). Callers that have a single
query may pass a ``(D,)`` or ``(1, D)`` vector.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:  # avoid importing torch-backed modules at runtime
    from src.datasets.base import PairKey, TemporalDataset
    from src.encoders.base import ImageTextEncoder

# Default sweep — a stable pair's Δ ≈ 0, so even a small positive threshold should
# drive the FPR to 0 if the zero_shot scorer is seasonally robust.
DEFAULT_THRESHOLDS: tuple = (0.0, 0.02, 0.05, 0.10)


def stable_pairs(dataset: "TemporalDataset") -> List["PairKey"]:
    """Return the pairs the dataset labels as stable (``PairLabel.stable``)."""
    out: List["PairKey"] = []
    for pair in dataset.list_pairs():
        label = dataset.get_pair_label(pair)
        if label is not None and label.stable:
            out.append(pair)
    return out


def false_positive_rate(scores: Sequence[float], threshold: float) -> float:
    """Fraction of stable-pair Δ-scores that exceed ``threshold`` (NaN if empty)."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr > threshold))


def fpr_sweep(scores: Sequence[float],
              thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> Dict[str, float]:
    """Map each threshold (formatted to 3 dp) to its false-positive rate."""
    return {f"{float(t):.3f}": false_positive_rate(scores, float(t)) for t in thresholds}


class ImageLevelChangeGate:
    """Whole-image ``zero_shot`` Δ-similarity change gate (binary, threshold-swept)."""

    def __init__(self, encoder: "ImageTextEncoder") -> None:
        self.encoder = encoder

    def delta_scores(
        self,
        dataset: "TemporalDataset",
        pairs: Sequence["PairKey"],
        query_vecs: np.ndarray,
    ) -> np.ndarray:
        """Per-pair Δ-similarity ``max_q cos(q, f_T2) − cos(q, f_T1)``.

        ``query_vecs`` is ``(Q, D)`` (or ``(D,)``), L2-normalised per the encoder
        contract; image embeddings come from ``encoder.encode_image`` (also
        L2-normalised), so cosines are plain dot products.
        """
        q = np.atleast_2d(np.asarray(query_vecs, dtype=np.float32))  # (Q, D)
        pairs = list(pairs)
        if not pairs:
            return np.empty(0, dtype=np.float32)
        # Collect all T1/T2 tiles first and encode each side in one batched pass
        # (the encoder batches internally), instead of a batch-of-2 GPU call per
        # pair. Per-image embeddings are unchanged (the ViT towers encode each
        # image independently), so this is numerically identical to the per-pair
        # loop, just far fewer kernel launches over a stable subset in the hundreds.
        imgs_t1, imgs_t2 = [], []
        for pair in pairs:
            im1, im2 = dataset.load_pair_images(pair)
            imgs_t1.append(im1)
            imgs_t2.append(im2)
        f1 = np.asarray(self.encoder.encode_image(imgs_t1), dtype=np.float32)  # (N, D)
        f2 = np.asarray(self.encoder.encode_image(imgs_t2), dtype=np.float32)  # (N, D)
        delta = f2 @ q.T - f1 @ q.T          # (N, Q)
        return delta.max(axis=1).astype(np.float32)


def evaluate_seasonal_fpr(
    dataset: "TemporalDataset",
    encoder: "ImageTextEncoder",
    query_vecs: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    pairs: Optional[Sequence["PairKey"]] = None,
) -> Dict[str, object]:
    """Run the gate over the stable subset and summarise mean Δ + FPR-vs-threshold.

    Returns a JSON-serialisable dict: stable-pair count, mean/std Δ-similarity, and
    ``fpr_by_threshold``. If ``pairs`` is omitted, the dataset's stable pairs are used.
    """
    pairs = list(pairs) if pairs is not None else stable_pairs(dataset)
    scores = ImageLevelChangeGate(encoder).delta_scores(dataset, pairs, query_vecs)
    has = scores.size > 0
    return {
        "n_stable_pairs": len(pairs),
        "mean_delta_similarity": (float(np.mean(scores)) if has else float("nan")),
        "std_delta_similarity": (float(np.std(scores)) if has else float("nan")),
        "fpr_by_threshold": fpr_sweep(scores, thresholds),
    }
