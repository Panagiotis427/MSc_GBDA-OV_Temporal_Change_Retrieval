"""Unit tests for the LEVIR-MCI loader (LEVIR-CC + building/road masks), on a
tiny synthetic fixture (no network / no 2.7 GB download)."""
import json

import numpy as np
import pytest
from PIL import Image

from src.datasets.levir_mci import MASK_VALUE, QUERY_TO_MASK_CLASS, LevirMCIDataset
from src.datasets.registry import build_dataset, list_datasets
from src.queries import get_queries


def _make_fixture(root):
    caps = {"images": [
        {"split": "test", "filename": "test_000000.png", "changeflag": 1,
         "sentences": [{"raw": "many buildings are built and a new road appears"}]},
        {"split": "test", "filename": "test_000001.png", "changeflag": 0,
         "sentences": [{"raw": "the scene is almost the same as before"}]},
    ]}
    (root / "LevirCCcaptions.json").write_text(json.dumps(caps), encoding="utf-8")
    for ab in ("A", "B"):
        d = root / "images" / "test" / ab
        d.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            arr = np.random.default_rng(i).integers(0, 255, (8, 8, 3)).astype("uint8")
            Image.fromarray(arr).save(d / f"test_00000{i}.png")
    # masks: 0 bg, 128 road, 255 building (grayscale, 3-channel like the real set)
    md = root / "images" / "test" / "label"
    md.mkdir(parents=True, exist_ok=True)
    m0 = np.zeros((8, 8), dtype="uint8")
    m0[0:3, 0:3] = MASK_VALUE["building"]   # building block
    m0[5:7, 5:7] = MASK_VALUE["road"]       # road block
    Image.fromarray(np.stack([m0] * 3, -1)).save(md / "test_000000.png")
    Image.fromarray(np.zeros((8, 8, 3), "uint8")).save(md / "test_000001.png")


@pytest.fixture
def ds(tmp_path):
    _make_fixture(tmp_path)
    return LevirMCIDataset(root=tmp_path, split="test")


def test_registered():
    assert "levir_mci" in list_datasets()
    assert get_queries("levir_mci"), "levir_mci queries not registered"


def test_inherits_cc_pairs(ds):
    assert len(ds.list_pairs()) == 2
    assert ds.name == "levir_mci"


def test_load_class_index_mask(ds):
    p = ds.list_pairs()[0]
    m = ds.load_change_mask(p, None)
    assert m.dtype == np.uint8
    assert set(np.unique(m)).issubset({0, 1, 2})
    assert (m == 2).sum() == 9      # 3x3 building block
    assert (m == 1).sum() == 4      # 2x2 road block


def test_load_boolean_class_mask(ds):
    p = ds.list_pairs()[0]
    bm = ds.load_change_mask(p, "building")
    rm = ds.load_change_mask(p, "road")
    assert bm.dtype == bool and bm.sum() == 9
    assert rm.sum() == 4
    assert not (bm & rm).any()      # disjoint


def test_no_change_mask_is_empty(ds):
    p = next(p for p in ds.list_pairs() if p.location_id == "test_000001")
    assert not ds.load_change_mask(p, "building").any()


def test_query_to_class_map_matches_mask_values():
    assert set(QUERY_TO_MASK_CLASS.values()) == set(MASK_VALUE)


def test_unknown_class_raises(ds):
    with pytest.raises(ValueError):
        ds.load_change_mask(ds.list_pairs()[0], "vegetation")


def test_build_via_registry(tmp_path):
    _make_fixture(tmp_path)
    ds = build_dataset("levir_mci", root=str(tmp_path), split="test")
    assert ds.name == "levir_mci"
    assert len(ds.list_pairs()) == 2
