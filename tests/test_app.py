"""
Headless test of the rewired app core (``SemanticChangeSearch.query``) on the
synthetic DEN fixture with a mock encoder — no browser, no CLIP. Verifies that
a query returns ranked real change events (actual T1/T2 tiles + heatmap) and
that the seasonal-vs-permanent note is label-grounded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.datasets.registry import get_dataset
from src.embeddings import compute_pair_embeddings
from src.app import RunConfig, SemanticChangeSearch

FIXTURE = Path("tests/fixtures/den_tiny")
from _mocks import MockEncoderBase, _CLASSES

_KW = {"buildings": "impervious_surface", "deforestation": "soil",
       "soil": "soil", "water": "water"}


class MockEncoder(MockEncoderBase):
    def encode_text(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            v = np.zeros(self.embed_dim, np.float32)
            for kw, c in _KW.items():
                if kw in t.lower():
                    v[_CLASSES.index(c)] += 1.0
            nrm = np.linalg.norm(v)
            out.append(v / nrm if nrm else v)
        return np.stack(out)


@pytest.fixture(scope="module")
def engine():
    if not FIXTURE.exists():
        pytest.skip("fixture missing")
    ds = get_dataset("dynamic_earthnet", root=str(FIXTURE),
                     pairing_strategy="bimonthly")
    enc = MockEncoder()
    store = compute_pair_embeddings(ds, enc)
    cfg = RunConfig(dataset="dynamic_earthnet", root=str(FIXTURE))
    return SemanticChangeSearch.from_components(ds, enc, store, cfg)


def test_query_returns_ranked_real_events(engine):
    evs = engine.query("deforestation forest cleared to bare soil",
                        approach="zero_shot", top_k=3)
    assert len(evs) == 3
    top = evs[0]
    assert top.location == "2065"
    assert top.t1_key == "2018-05-01" and top.t2_key == "2018-07-01"
    assert isinstance(top.t1_img, Image.Image)
    assert isinstance(top.t2_img, Image.Image)
    assert isinstance(top.heatmap, Image.Image)          # heatmap rendered
    assert 0.0 <= top.confidence <= 1.0
    assert top.seasonal_note == "permanent land-cover change"
    # ranks monotonic, scores descending
    assert [e.rank for e in evs] == [1, 2, 3]
    assert evs[0].score >= evs[1].score >= evs[2].score


def test_seasonal_pair_flagged(engine):
    snow_pair = next(
        p for p in engine.dataset.list_pairs()
        if engine.dataset.get_pair_label(p).dominant_t1_class == "snow_and_ice"
    )
    _, note = engine._describe(snow_pair)
    assert "SEASONAL" in note


def test_peft_without_adapter_errors(engine):
    with pytest.raises(RuntimeError, match="no adapter"):
        engine.query("anything", approach="peft", top_k=2)


def test_dataset_profiles_contract():
    """The in-app Dataset dropdown must only offer corpora that actually load
    (a launch profile + a registered query set), with colour pinned correctly."""
    import os
    from src.app import DATASET_PROFILES, _app_dataset_choices
    from src.queries import get_queries

    for ds, prof in DATASET_PROFILES.items():
        assert get_queries(ds), f"{ds} has no query set"
        assert os.path.isabs(prof["root"]), f"{ds} root not absolute"
    # DEN honours the colour dropdown (no pinned colour); other corpora are rgb-only
    assert "color_mode" not in DATASET_PROFILES["dynamic_earthnet"]
    for ds in ("levir_cc", "levir_mci", "second_cc"):
        assert DATASET_PROFILES[ds]["color_mode"] == "rgb"
    # dropdown = query-set ∩ profile; QFabric (needs extra loader args) is excluded
    choices = set(_app_dataset_choices())
    assert choices == set(DATASET_PROFILES)
    assert "qfabric_teo" not in choices and "qfabric_status" not in choices
