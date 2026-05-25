"""Tests for src.datasets.qfabric.QFabricDataset (dict-injection mode)."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.datasets.qfabric import QFabricDataset
from src.datasets.base import PairKey


def _make_fake_dataset(n_locs: int = 4, n_times: int = 5, dim: int = 8):
    """Return (embedding_lookup, metadata_df) for QFabricDataset injection."""
    emb = {
        f"loc_{i}": np.random.randn(n_times, dim).astype(np.float32)
        for i in range(n_locs)
    }
    rows = []
    base = pd.Timestamp("2023-01-01")
    for loc, arr in emb.items():
        for t in range(n_times):
            rows.append({
                "location": loc,
                "timestamp": base + pd.Timedelta(days=30 * t),
                "timepoint_idx": t,
                "t_key": f"t{t+1}",
                "pair_id": f"{loc}_pair_{t // 2}",
                "dataset_name": "qfabric",
            })
    df = pd.DataFrame(rows)
    return emb, df


class TestQFabricDatasetDictMode:
    def test_construction(self):
        emb, df = _make_fake_dataset()
        ds = QFabricDataset(embedding_lookup=emb, metadata_df=df)
        assert ds.name == "qfabric"

    def test_list_locations(self):
        emb, df = _make_fake_dataset(n_locs=3)
        ds = QFabricDataset(embedding_lookup=emb, metadata_df=df)
        locs = ds.list_locations()
        assert len(locs) == 3
        assert all(isinstance(l, str) for l in locs)

    def test_list_pairs(self):
        emb, df = _make_fake_dataset(n_locs=2, n_times=5)
        ds = QFabricDataset(embedding_lookup=emb, metadata_df=df)
        pairs = ds.list_pairs()
        # 5 timepoints → 4 consecutive pairs per location × 2 locs = 8
        assert len(pairs) == 8

    def test_pair_keys_are_pair_key_instances(self):
        emb, df = _make_fake_dataset(n_locs=2, n_times=5)
        ds = QFabricDataset(embedding_lookup=emb, metadata_df=df)
        for pk in ds.list_pairs():
            assert isinstance(pk, PairKey)

    def test_load_metadata_required_cols(self):
        emb, df = _make_fake_dataset()
        ds = QFabricDataset(embedding_lookup=emb, metadata_df=df)
        meta = ds.load_metadata()
        for col in ("location", "timestamp", "t_key", "pair_id", "dataset_name"):
            assert col in meta.columns

    def test_get_pair_label_returns_none_or_label(self):
        emb, df = _make_fake_dataset(n_locs=2, n_times=5)
        ds = QFabricDataset(embedding_lookup=emb, metadata_df=df)
        pairs = ds.list_pairs()
        label = ds.get_pair_label(pairs[0])
        # Without change_type column in df → None is acceptable
        assert label is None or hasattr(label, "stable")

    def test_embedding_lookup_property(self):
        emb, df = _make_fake_dataset(n_locs=2)
        ds = QFabricDataset(embedding_lookup=emb, metadata_df=df)
        el = ds.embedding_lookup
        assert isinstance(el, dict)
        assert set(el.keys()) == set(emb.keys())
