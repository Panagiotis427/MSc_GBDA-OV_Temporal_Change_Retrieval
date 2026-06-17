"""
Fast tests for the embedding cache reuse/recompute logic in
``src.embeddings.load_or_compute`` (no CLIP, no network — uses the deterministic
mock encoder + the synthetic DEN fixture).

Covers the previously-untested branches:
  - a valid cache is reused (pair set matches),
  - a *stale* cache (on-disk pair set != the dataset's) is detected and recomputed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.datasets.registry import get_dataset
from src.embeddings import (
    PairEmbeddingStore,
    cache_path,
    cache_tag_for,
    compute_pair_embeddings,
    load_or_compute,
)

from _mocks import MockEncoderBase

FIXTURE = Path("tests/fixtures/den_tiny")


@pytest.fixture(scope="module")
def den():
    if not FIXTURE.exists():
        pytest.skip("DEN fixture missing; run scripts.make_den_fixture")
    return get_dataset("dynamic_earthnet", root=str(FIXTURE),
                       pairing_strategy="bimonthly")


def _pairs(store_or_ds):
    return [tuple(p) for p in store_or_ds.pairs] if hasattr(store_or_ds, "pairs") \
        else [tuple(p) for p in store_or_ds.list_pairs()]


def test_load_or_compute_reuses_valid_cache(den, tmp_path):
    enc = MockEncoderBase()
    tag = cache_tag_for("all", "rgb")
    full = compute_pair_embeddings(den, enc)
    full.save(cache_path(tmp_path, den.name, enc.name, tag))

    got = load_or_compute(den, enc, cache_dir=tmp_path, cache_tag=tag)
    assert _pairs(got) == _pairs(den)
    assert len(got) == len(full)


def test_load_or_compute_recomputes_on_stale_pairset(den, tmp_path):
    enc = MockEncoderBase()
    tag = cache_tag_for("all", "rgb")
    full = compute_pair_embeddings(den, enc)
    assert len(full) >= 2, "fixture needs >=2 pairs to drop one"

    # Persist a STALE cache missing the last pair, so the on-disk pair set differs
    # from what the dataset reports -> load_or_compute must recompute, not reuse.
    stale = PairEmbeddingStore(
        dataset_name=full.dataset_name, encoder_name=full.encoder_name,
        embed_dim=full.embed_dim, pairs=full.pairs[:-1],
        f_t1=full.f_t1[:-1], f_t2=full.f_t2[:-1],
    )
    path = cache_path(tmp_path, den.name, enc.name, tag)
    stale.save(path)

    got = load_or_compute(den, enc, cache_dir=tmp_path, cache_tag=tag)
    assert len(got) == len(den.list_pairs())   # recomputed over the full pair set
    assert _pairs(got) == _pairs(den)
