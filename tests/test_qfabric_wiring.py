"""QFabric structural wiring: images-only parquet loading via build_dataset,
and the registry opts dropping generic kwargs (color_mode/pairing/split) that
the loader doesn't accept. No network — a tiny synthetic shard is written to
tmp_path mirroring the real EVER-Z/QFabric_mt_images_1024 schema.
"""
from __future__ import annotations

import io

import pandas as pd
import pytest
from PIL import Image

from src.datasets.registry import build_dataset


def _png_bytes(color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, format="PNG")
    return buf.getvalue()


def _write_tiny_shard(path, n_locs=3):
    """One parquet row per location, 5 image structs + 5 name cols (real schema)."""
    rows = []
    for li in range(n_locs):
        row = {}
        for ti in range(5):
            row[f"t{ti+1}_image"] = {"bytes": _png_bytes((li * 40, ti * 40, 80)),
                                     "path": None}
            row[f"t{ti+1}_image_name"] = f"{li}.d{ti+1}.0202201{ti}_0_1024.tif"
        rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


@pytest.fixture
def shard_dir(tmp_path):
    _write_tiny_shard(tmp_path / "train-00000-of-00597.parquet", n_locs=3)
    return tmp_path


def test_build_dataset_drops_generic_kwargs(shard_dir):
    # Previously color_mode/pairing/split crashed QFabricDataset.__init__.
    ds = build_dataset("qfabric", root=str(shard_dir),
                       color_mode="rgb", pairing="bimonthly", split="test")
    assert ds.name == "qfabric"
    assert len(ds.list_locations()) == 3


def test_images_only_pairs_and_images(shard_dir):
    ds = build_dataset("qfabric", root=str(shard_dir), color_mode="nrg")
    pairs = ds.list_pairs()
    assert len(pairs) == 3 * 4  # 5 timepoints -> 4 consecutive pairs per location
    t1, t2 = ds.load_pair_images(pairs[0])
    assert isinstance(t1, Image.Image) and t1.size == (16, 16)
    assert ds.get_pair_label(pairs[0]) is None  # labels skipped by design


def test_metadata_has_required_columns(shard_dir):
    ds = build_dataset("qfabric", root=str(shard_dir))
    meta = ds.load_metadata()
    for col in ("location", "timestamp", "t_key", "pair_id", "dataset_name"):
        assert col in meta.columns


def test_end_to_end_with_mock_encoder(shard_dir):
    """Images-only dataset flows through the standard embeddings + retrieval path."""
    import numpy as np
    from src.embeddings import compute_pair_embeddings
    from src.retrieval import ChangeRetriever

    class _Enc:
        name = "mock"; embed_dim = 4
        import torch as _t
        device = _t.device("cpu")
        def encode_image(self, images, batch_size=32):
            return np.ones((len(images), 4), dtype=np.float32)
        def encode_text(self, texts, batch_size=32):
            t = [texts] if isinstance(texts, str) else texts
            return np.ones((len(t), 4), dtype=np.float32)

    ds = build_dataset("qfabric", root=str(shard_dir))
    store = compute_pair_embeddings(ds, _Enc())
    assert len(store) == 12 and store.embed_dim == 4
    r = ChangeRetriever(store, _Enc())
    res = r.search("new construction", approach="zero_shot", top_k=3)
    assert len(res) == 3
