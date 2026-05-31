"""TEOChatlas-QFabric loader + query set — synthetic .tif fixture, no network/GPU."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from src.datasets.registry import build_dataset
from src.datasets.qfabric_teo import parse_crop


def _write_fixture(tmp_path):
    imgs = tmp_path / "QFabric"
    imgs.mkdir()
    spec = {"100_0_256": "residential", "200_0_256": "road",
            "300_0_256": "commercial"}
    for ck, _ct in spec.items():
        loc, xoff, yoff = ck.split("_")
        for n in (1, 2):
            Image.new("RGB", (16, 16), (n * 50, 60, 70)).save(
                imgs / f"{loc}.d{n}.0101201{n}_{xoff}_{yoff}.tif")
    labels = tmp_path / "qfabric_teo_labels.json"
    json.dump(spec, open(labels, "w"))
    return str(imgs), str(labels), spec


def test_parse_crop():
    ck, loc, day, date = parse_crop("300.d2.12262015_256_4096.tif")
    assert ck == "300_256_4096" and loc == "300" and day == "d2"


def test_loader_pairs_and_labels(tmp_path):
    root, labels, spec = _write_fixture(tmp_path)
    ds = build_dataset("qfabric_teo", root=root, labels_path=labels,
                       color_mode="rgb", pairing="bimonthly", split="x")  # generic kwargs dropped
    assert ds.name == "qfabric_teo"
    assert len(ds.list_locations()) == 3
    pairs = ds.list_pairs()
    assert len(pairs) == 3  # 2 timepoints -> 1 pair per crop
    for p in pairs:
        lb = ds.get_pair_label(p)
        assert lb is not None and lb.change_type == spec[p.location_id]
    t1, t2 = ds.load_pair_images(pairs[0])
    assert isinstance(t1, Image.Image) and t1.size == (16, 16)


def test_metadata_columns(tmp_path):
    root, labels, _ = _write_fixture(tmp_path)
    ds = build_dataset("qfabric_teo", root=root, labels_path=labels)
    meta = ds.load_metadata()
    for col in ("location", "timestamp", "t_key", "pair_id", "dataset_name"):
        assert col in meta.columns


def test_subsample_per_class(tmp_path):
    root, labels, _ = _write_fixture(tmp_path)
    ds = build_dataset("qfabric_teo", root=root, labels_path=labels, max_per_class=1)
    # 3 distinct classes, 1 each -> 3 crops (already 1/class here)
    assert len(ds.list_locations()) == 3


def test_queries_registered_and_discriminative():
    from src.queries import get_queries
    from src.datasets.base import PairLabel
    qs = get_queries("qfabric_teo")
    assert len(qs) == 6
    res_q = next(q for q in qs if "residential" in q.text)
    assert res_q.predicate(PairLabel(change_type="residential", stable=False)) is True
    assert res_q.predicate(PairLabel(change_type="road", stable=False)) is False


def test_end_to_end_benchmark(tmp_path):
    from src.embeddings import compute_pair_embeddings
    from src.retrieval import ChangeRetriever
    from src.benchmark import run_benchmark

    root, labels, _ = _write_fixture(tmp_path)
    ds = build_dataset("qfabric_teo", root=root, labels_path=labels)

    class _Enc:
        name = "mock"; embed_dim = 4
        import torch as _t
        device = _t.device("cpu")
        def encode_image(self, images, batch_size=32):
            return np.ones((len(images), 4), dtype=np.float32)
        def encode_text(self, texts, batch_size=32):
            t = [texts] if isinstance(texts, str) else texts
            return np.ones((len(t), 4), dtype=np.float32)

    store = compute_pair_embeddings(ds, _Enc())
    rep = run_benchmark(ds, ChangeRetriever(store, _Enc()), approach="zero_shot")
    assert rep.per_query  # at least one query has positives in the fixture
    assert 0.0 <= rep.mAP <= 1.0
