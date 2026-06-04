"""Unit tests for the image-level seasonal-robustness gate (`src.seasonal_gate`).

No network / no CLIP weights: a solid-colour fake DEN dataset feeds the shared
``MockEncoderBase`` (palette-majority one-hot ``encode_image``), so a stable pair
(identical T1/T2 tiles) yields Δ-similarity exactly 0 and a change pair yields a
known Δ. The tests pin: the FPR math + monotonicity, stable-pair selection via
``PairLabel.stable``, the ``zero_shot`` Δ formula, and the end-to-end summary.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.datasets._palette import DEN_CLASS_NAMES, DEN_PALETTE
from src.datasets.base import PairKey, PairLabel
from src.seasonal_gate import (
    ImageLevelChangeGate,
    evaluate_seasonal_fpr,
    false_positive_rate,
    fpr_sweep,
    stable_pairs,
)
from tests._mocks import MockEncoderBase

_DIM = len(DEN_CLASS_NAMES)


def _tile(cls: str) -> Image.Image:
    color = np.array(DEN_PALETTE[cls], dtype=np.uint8)
    return Image.fromarray(np.broadcast_to(color, (8, 8, 3)).copy())


def _onehot(cls: str) -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float32)
    v[DEN_CLASS_NAMES.index(cls)] = 1.0
    return v


class _FakeDEN:
    """Minimal ``TemporalDataset`` of solid-colour tiles. spec: (loc, c1, c2, stable)."""

    name = "dynamic_earthnet"
    temporal_axis_type = "daily"

    def __init__(self, spec):
        self._spec = {loc: (c1, c2, stable) for (loc, c1, c2, stable) in spec}
        self._order = [loc for (loc, _, _, _) in spec]

    def list_pairs(self):
        return [PairKey(loc, "t1", "t2") for loc in self._order]

    def get_pair_label(self, pair):
        c1, c2, stable = self._spec[pair.location_id]
        return PairLabel(
            change_type="stable" if stable else f"{c1}->{c2}",
            stable=stable, dominant_t1_class=c1, dominant_t2_class=c2,
        )

    def load_pair_images(self, pair):
        c1, c2, _ = self._spec[pair.location_id]
        return _tile(c1), _tile(c2)


# --- FPR math -------------------------------------------------------------

def test_false_positive_rate_exact_and_empty():
    scores = [0.0, 0.03, 0.06, 0.12]
    assert false_positive_rate(scores, 0.0) == 0.75   # strict > : 3 of 4
    assert false_positive_rate(scores, 0.05) == 0.5
    assert false_positive_rate(scores, 0.10) == 0.25
    assert np.isnan(false_positive_rate([], 0.0))


def test_fpr_sweep_monotone_non_increasing():
    scores = [0.0, 0.03, 0.06, 0.12]
    thresholds = [0.0, 0.02, 0.05, 0.10]
    sweep = fpr_sweep(scores, thresholds)
    assert list(sweep.keys()) == ["0.000", "0.020", "0.050", "0.100"]
    vals = list(sweep.values())
    assert all(a >= b for a, b in zip(vals, vals[1:])), f"FPR not monotone: {vals}"


# --- stable-pair selection ------------------------------------------------

def test_stable_pairs_selects_only_stable():
    ds = _FakeDEN([
        ("a", "water", "water", True),
        ("b", "water", "impervious_surface", False),
        ("c", "forest_and_other_vegetation", "forest_and_other_vegetation", True),
    ])
    selected = {p.location_id for p in stable_pairs(ds)}
    assert selected == {"a", "c"}


# --- zero_shot Δ formula --------------------------------------------------

def test_delta_formula_matches_zero_shot():
    # Change pair impervious(T1) -> water(T2). MockEncoder returns one-hot of the
    # dominant class, so Δ = cos(q, onehot(T2)) - cos(q, onehot(T1)).
    ds = _FakeDEN([("a", "impervious_surface", "water", False)])
    gate = ImageLevelChangeGate(MockEncoderBase())
    pairs = ds.list_pairs()

    q_t2 = _onehot("water")
    assert gate.delta_scores(ds, pairs, q_t2)[0] == 1.0          # 1 - 0
    q_t1 = _onehot("impervious_surface")
    assert gate.delta_scores(ds, pairs, q_t1)[0] == -1.0         # 0 - 1
    # max over both queries -> the firing query wins.
    both = np.stack([q_t1, q_t2])
    assert gate.delta_scores(ds, pairs, both)[0] == 1.0


# --- end-to-end summary ---------------------------------------------------

def test_stable_pairs_score_zero_and_fpr_zero():
    ds = _FakeDEN([
        ("a", "water", "water", True),
        ("b", "impervious_surface", "impervious_surface", True),
    ])
    summary = evaluate_seasonal_fpr(
        ds, MockEncoderBase(), _onehot("water"), thresholds=[0.0, 0.02, 0.05])
    assert summary["n_stable_pairs"] == 2
    assert summary["mean_delta_similarity"] == 0.0      # identical T1/T2 -> Δ == 0
    assert all(v == 0.0 for v in summary["fpr_by_threshold"].values())


def test_evaluate_defaults_to_dataset_stable_subset():
    # One change pair is present but must be excluded (only stable pairs scored).
    ds = _FakeDEN([
        ("a", "water", "water", True),
        ("b", "water", "impervious_surface", False),
        ("c", "soil", "soil", True),
    ])
    summary = evaluate_seasonal_fpr(ds, MockEncoderBase(), _onehot("water"))
    assert summary["n_stable_pairs"] == 2
