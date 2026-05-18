"""Tests for src.datasets.base — protocol, PairKey, PairLabel, helpers."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.datasets.base import (
    PairKey,
    PairLabel,
    metadata_from_dataset,
    pair_iter_from_dataset,
)


# ---------------------------------------------------------------------------
# Minimal concrete implementation for tests
# ---------------------------------------------------------------------------

class _FakeDataset:
    name = "fake"
    temporal_axis_type = "fixed-5"

    def __init__(self, n_locs: int = 3, n_times: int = 2):
        self._locs = [f"loc_{i}" for i in range(n_locs)]
        self._n_times = n_times
        self._pairs = [
            PairKey(loc, f"t{t}", f"t{t+1}")
            for loc in self._locs
            for t in range(n_times - 1)
        ]

    def list_locations(self):
        return list(self._locs)

    def list_pairs(self):
        return list(self._pairs)

    def load_image(self, location_id, t_key):
        return Image.new("RGB", (8, 8))

    def load_pair_images(self, pair):
        return self.load_image(pair.location_id, pair.t1_key), self.load_image(pair.location_id, pair.t2_key)

    def load_metadata(self):
        rows = []
        for loc in self._locs:
            for t in range(self._n_times):
                rows.append({
                    "location": loc,
                    "timestamp": pd.Timestamp(f"2023-0{t+1}-01"),
                    "t_key": f"t{t}",
                    "pair_id": f"{loc}_t{t}_t{t+1}" if t < self._n_times - 1 else f"{loc}_t{t-1}_t{t}",
                    "dataset_name": self.name,
                    "timepoint_idx": t,
                })
        return pd.DataFrame(rows)

    def get_pair_label(self, pair):
        return PairLabel(change_type="stable", stable=True)


# ---------------------------------------------------------------------------
# PairKey
# ---------------------------------------------------------------------------

class TestPairKey:
    def test_namedtuple_fields(self):
        pk = PairKey("loc1", "t1", "t2")
        assert pk.location_id == "loc1"
        assert pk.t1_key == "t1"
        assert pk.t2_key == "t2"

    def test_hashable(self):
        pk = PairKey("loc1", "t1", "t2")
        assert hash(pk) is not None
        s = {pk}
        assert pk in s

    def test_equality(self):
        assert PairKey("a", "b", "c") == PairKey("a", "b", "c")
        assert PairKey("a", "b", "c") != PairKey("a", "b", "d")


# ---------------------------------------------------------------------------
# PairLabel
# ---------------------------------------------------------------------------

class TestPairLabel:
    def test_defaults(self):
        label = PairLabel(change_type="stable", stable=True)
        assert label.dominant_t1_class is None
        assert label.class_change_mask_fraction == {}

    def test_full_construction(self):
        label = PairLabel(
            change_type="forest->impervious_surface",
            stable=False,
            dominant_t1_class="forest_and_other_vegetation",
            dominant_t2_class="impervious_surface",
            class_change_mask_fraction={"forest_and_other_vegetation": {"lost_fraction": 0.3}},
        )
        assert not label.stable
        assert label.dominant_t2_class == "impervious_surface"


# ---------------------------------------------------------------------------
# metadata_from_dataset
# ---------------------------------------------------------------------------

class TestMetadataFromDataset:
    def test_required_columns(self):
        ds = _FakeDataset(n_locs=2, n_times=3)
        df = metadata_from_dataset(ds)
        for col in ("location", "timestamp", "t_key", "pair_id", "dataset_name"):
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count(self):
        ds = _FakeDataset(n_locs=4, n_times=2)
        df = metadata_from_dataset(ds)
        assert len(df) == 4 * 2  # locs × times


# ---------------------------------------------------------------------------
# pair_iter_from_dataset
# ---------------------------------------------------------------------------

class TestPairIterFromDataset:
    def test_yields_correct_count(self):
        ds = _FakeDataset(n_locs=3, n_times=3)
        embed_dim = 16
        emb = {
            loc: np.random.randn(3, embed_dim).astype(np.float32)
            for loc in ds.list_locations()
        }
        results = list(pair_iter_from_dataset(ds, emb))
        # 3 locs × (n_times-1=2) pairs = 6
        assert len(results) == 6

    def test_yields_pair_key_and_embeddings(self):
        ds = _FakeDataset(n_locs=2, n_times=2)
        dim = 8
        emb = {loc: np.ones((2, dim), dtype=np.float32) for loc in ds.list_locations()}
        for pk, e1, e2 in pair_iter_from_dataset(ds, emb):
            assert isinstance(pk, PairKey)
            assert e1.shape == (dim,)
            assert e2.shape == (dim,)
