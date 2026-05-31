"""Offline smoke test for scripts/make_comparison_figure.py — mock encoder +
DEN fixture + a randomly-initialised ProjectionHead. Asserts a non-empty PNG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pytest

from src.datasets.registry import get_dataset
from src.embeddings import compute_pair_embeddings
from src.model import create_projection_head
from src.retrieval import ChangeRetriever
from scripts import make_comparison_figure as mcf

FIXTURE = Path("tests/fixtures/den_tiny")

from _mocks import _CLASSES  # noqa: E402
from test_change_retrieval import MockEncoder  # noqa: E402


@pytest.fixture(scope="module")
def den():
    if not FIXTURE.exists():
        pytest.skip("DEN fixture missing; run scripts.make_den_fixture")
    return get_dataset("dynamic_earthnet", root=str(FIXTURE),
                       pairing_strategy="bimonthly")


def test_render_writes_png(den, tmp_path):
    store = compute_pair_embeddings(den, MockEncoder())
    retr = ChangeRetriever(store, MockEncoder())
    adapter = create_projection_head(input_dim=len(_CLASSES),
                                     output_dim=len(_CLASSES), hidden_dims=(16,))
    p = mcf.render(den, retr, adapter, tmp_path, encoder="mock",
                   split="train", top_k=3)
    assert p is not None and p.exists() and p.stat().st_size > 0


def test_render_zero_shot_only_without_adapter(den, tmp_path):
    store = compute_pair_embeddings(den, MockEncoder())
    retr = ChangeRetriever(store, MockEncoder())
    p = mcf.render(den, retr, None, tmp_path, encoder="mock", split="train", top_k=2)
    assert p is not None and p.exists()


def test_representative_queries_picks_substrings():
    from src.queries import get_queries
    qs = representative = mcf.representative_queries(get_queries("dynamic_earthnet"))
    assert qs, "should pick at least one representative query"
    assert len(qs) <= 3
