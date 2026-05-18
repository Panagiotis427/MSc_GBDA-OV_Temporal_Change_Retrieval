"""Tests for src/rerank.py — diversity and coherence re-ranking strategies."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.datasets.base import PairKey
from src.rerank import Reranker, RERANK_STRATEGIES

_META = {
    "aoi_a": {"lat_c": 48.0, "lon_c": 2.0},    # Paris area
    "aoi_b": {"lat_c": 48.5, "lon_c": 2.3},    # ~60 km from aoi_a
    "aoi_c": {"lat_c": -33.9, "lon_c": 151.2}, # Sydney — far from both
}

# Two pairs from aoi_a, one each from aoi_b and aoi_c
_PAIRS = [
    PairKey("aoi_a", "t1", "t2"),
    PairKey("aoi_a", "t3", "t4"),
    PairKey("aoi_b", "t1", "t2"),
    PairKey("aoi_c", "t1", "t2"),
]
# Descending score: aoi_a/t1 best, then aoi_a/t3, aoi_b, aoi_c
_SCORES = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)


@pytest.fixture
def rr(tmp_path):
    p = tmp_path / "aoi_metadata.json"
    p.write_text(json.dumps(_META))
    return Reranker(p)


# -- strategy list ----------------------------------------------------------

def test_strategies_constant():
    assert "diversity" in RERANK_STRATEGIES
    assert "coherence" in RERANK_STRATEGIES


def test_unknown_strategy_raises(rr):
    with pytest.raises(ValueError, match="Unknown rerank strategy"):
        rr.rerank(_SCORES, _PAIRS, top_k=2, strategy="magic")


# -- diversity --------------------------------------------------------------

def test_diversity_top1_highest_score(rr):
    order = rr.rerank(_SCORES, _PAIRS, top_k=3, strategy="diversity")
    assert _PAIRS[order[0]].location_id == "aoi_a"


def test_diversity_second_result_different_location(rr):
    order = rr.rerank(_SCORES, _PAIRS, top_k=3, strategy="diversity")
    assert _PAIRS[order[1]].location_id != "aoi_a"


def test_diversity_fills_remaining_with_repeats(rr):
    # top_k=4 forces a repeat location in last slot
    order = rr.rerank(_SCORES, _PAIRS, top_k=4, strategy="diversity")
    assert len(order) == 4
    locs = [_PAIRS[i].location_id for i in order]
    # aoi_a must appear twice (only way to fill 4 slots with 3 unique locs + 4 pairs)
    assert locs.count("aoi_a") == 2


def test_diversity_skips_inf_scores(rr):
    scores = np.array([0.9, -np.inf, 0.7, 0.6], dtype=np.float32)
    order = rr.rerank(scores, _PAIRS, top_k=3, strategy="diversity")
    for i in order:
        assert np.isfinite(scores[i])


# -- coherence --------------------------------------------------------------

def test_coherence_top1_is_highest_cosine(rr):
    order = rr.rerank(_SCORES, _PAIRS, top_k=3, strategy="coherence", geo_weight=0.3)
    assert _PAIRS[order[0]].location_id == "aoi_a"


def test_coherence_nearby_ranked_above_far(rr):
    # With geo_weight=0.9, proximity dominates; aoi_b (near aoi_a) should beat aoi_c.
    # Use top_k=4 so all pairs are ranked (high geo_weight can push aoi_c out of top-3).
    order = rr.rerank(_SCORES, _PAIRS, top_k=4, strategy="coherence", geo_weight=0.9)
    locs = [_PAIRS[i].location_id for i in order]
    b_rank = next(i for i, l in enumerate(locs) if l == "aoi_b")
    c_rank = next(i for i, l in enumerate(locs) if l == "aoi_c")
    assert b_rank < c_rank


def test_coherence_all_inf_returns_empty(rr):
    scores = np.full(4, -np.inf, dtype=np.float32)
    order = rr.rerank(scores, _PAIRS, top_k=2, strategy="coherence")
    assert len(order) == 0


def test_coherence_inf_masked_excluded(rr):
    scores = np.array([0.9, -np.inf, 0.7, 0.6], dtype=np.float32)
    order = rr.rerank(scores, _PAIRS, top_k=2, strategy="coherence")
    for i in order:
        assert np.isfinite(scores[i])


# -- output shape -----------------------------------------------------------

def test_output_length_bounded_by_top_k(rr):
    for k in (1, 2, 3, 4):
        order = rr.rerank(_SCORES, _PAIRS, top_k=k, strategy="diversity")
        assert len(order) <= k


def test_output_indices_in_range(rr):
    order = rr.rerank(_SCORES, _PAIRS, top_k=3, strategy="diversity")
    assert all(0 <= i < len(_PAIRS) for i in order)
