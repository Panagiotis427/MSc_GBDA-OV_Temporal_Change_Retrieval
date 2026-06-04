"""`labels_index.parquet` round-trip for the raster (.tif) DEN loader.

Guards the index fast-path: ``build_label_index`` persists the per-class change
fractions, and ``DENDataset.get_pair_label`` serves them straight from the index
— no label rasters required. This is the property that lets fraction-based
relevance predicates (``benchmark._gained`` / ``_lost``) work off the index
without silently matching nothing, and without re-reading rasters per pair. The
second test covers the back-compat fallback: an older index (no fraction column)
still loads and derives fractions from the rasters, with a one-time warning.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from scripts.download_den import build_label_index
from src.datasets.dynamic_earthnet import FRAC_JSON_COL, DENDataset

FIXTURE = Path("tests/fixtures/den_tiny")


def _labels_with_fractions(ds):
    labels = [lb for lb in map(ds.get_pair_label, ds.list_pairs()) if lb is not None]
    with_frac = [lb for lb in labels if lb.class_change_mask_fraction]
    return labels, with_frac


def _assert_fraction_shape(label):
    _cls, frac = next(iter(label.class_change_mask_fraction.items()))
    assert set(frac) == {"gained_fraction", "lost_fraction"}
    assert 0.0 <= frac["gained_fraction"] <= 1.0
    assert 0.0 <= frac["lost_fraction"] <= 1.0


@pytest.mark.skipif(not FIXTURE.exists(),
                    reason="DEN fixture missing; run scripts.make_den_fixture")
def test_index_carries_fractions_and_serves_raster_free(tmp_path):
    root = tmp_path / "den"
    shutil.copytree(FIXTURE, root)
    # Rebuild the index from rasters so it gets the current schema.
    (root / "labels_index.parquet").unlink(missing_ok=True)

    out = build_label_index(root)
    assert FRAC_JSON_COL in pd.read_parquet(out).columns

    # Remove the label rasters: get_pair_label must now serve labels (and their
    # per-class change fractions) purely from the index, proving the fast path
    # no longer depends on raster I/O.
    shutil.rmtree(root / "labels")
    labels, with_frac = _labels_with_fractions(DENDataset(root))
    assert labels, "no labels served from the index after removing rasters"
    assert with_frac, "index served no per-class change fractions"
    _assert_fraction_shape(with_frac[0])


@pytest.mark.skipif(not FIXTURE.exists(),
                    reason="DEN fixture missing; run scripts.make_den_fixture")
def test_old_index_without_fraction_column_falls_back_to_rasters(tmp_path):
    root = tmp_path / "den"
    shutil.copytree(FIXTURE, root)
    (root / "labels_index.parquet").unlink(missing_ok=True)
    out = build_label_index(root)

    # Simulate an index built before the fraction column existed.
    df = pd.read_parquet(out).drop(columns=[FRAC_JSON_COL])
    df.to_parquet(out, index=False)

    # Rasters are still present, so the fallback derives fractions — and the
    # loader warns once that the index is stale.
    with pytest.warns(UserWarning, match=FRAC_JSON_COL):
        ds = DENDataset(root)
    labels, with_frac = _labels_with_fractions(ds)
    assert labels, "no labels served from the stale index"
    assert with_frac, "raster fallback produced no fractions"
    _assert_fraction_shape(with_frac[0])
