"""
Deterministic end-to-end test of the change-retrieval core (P2/P3) using a
mock encoder + the synthetic DEN fixture — no network, no CLIP.

A mock encoder maps each tile to a one-hot over DEN classes (by dominant
palette colour) and maps a query to the one-hot of its *destination* class.
Under Δ-similarity (``zero_shot``) the pair whose T2 introduces that class
must rank first, which lets us assert exact Recall@1 / AP on engineered
fixture transitions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.datasets.registry import get_dataset
from src.embeddings import PairEmbeddingStore, compute_pair_embeddings
from src.retrieval import ChangeRetriever
from src.benchmark import Query, run_benchmark, _transition
from src.queries.den import AGRI, IMPERVIOUS, FOREST, SOIL

FIXTURE = Path("tests/fixtures/den_tiny")

from _mocks import MockEncoderBase, _CLASSES

_KEYWORDS = {  # query keyword -> destination ("after") class; unambiguous cues only
    "buildings": "impervious_surface", "urban": "impervious_surface",
    "soil": "soil", "deforestation": "soil",
    "water": "water", "flooding": "water",
}


class MockEncoder(MockEncoderBase):
    def encode_text(self, texts, batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            v = np.zeros(self.embed_dim, dtype=np.float32)
            for kw, cls in _KEYWORDS.items():
                if kw in t.lower():
                    v[_CLASSES.index(cls)] += 1.0
            n = np.linalg.norm(v)
            out.append(v / n if n > 0 else v)
        return np.stack(out)


@pytest.fixture(scope="module")
def den():
    if not FIXTURE.exists():
        pytest.skip("DEN fixture missing; run scripts.make_den_fixture")
    return get_dataset("dynamic_earthnet", root=str(FIXTURE),
                        pairing_strategy="bimonthly")


@pytest.fixture(scope="module")
def store(den):
    return compute_pair_embeddings(den, MockEncoder())


def test_store_shapes_and_roundtrip(store, tmp_path):
    assert len(store) == 6
    assert store.f_t1.shape == store.f_t2.shape == (6, len(_CLASSES))
    p = tmp_path / "s.npz"
    store.save(p)
    again = PairEmbeddingStore.load(p)
    assert again.pairs == store.pairs
    np.testing.assert_allclose(again.f_t1, store.f_t1)


def test_change_feature_modes(store):
    assert store.change_features("difference").shape == (6, len(_CLASSES))
    assert store.change_features("concatenate").shape == (6, 2 * len(_CLASSES))


def test_zero_shot_ranks_relevant_pair_first(den, store):
    r = ChangeRetriever(store, MockEncoder())
    # forest -> soil pair must top "deforestation ... soil"
    res = r.search("deforestation forest cleared to bare soil", approach="zero_shot", top_k=6)
    top = res[0].pair
    lb = den.get_pair_label(top)
    assert lb.dominant_t1_class == FOREST and lb.dominant_t2_class == SOIL
    assert res[0].score > res[-1].score

    res2 = r.search("new buildings on agricultural land", approach="zero_shot", top_k=6)
    lb2 = den.get_pair_label(res2[0].pair)
    assert lb2.dominant_t1_class == AGRI and lb2.dominant_t2_class == IMPERVIOUS


def test_benchmark_metrics_well_formed(den, store):
    r = ChangeRetriever(store, MockEncoder())
    rep = run_benchmark(den, r, approach="zero_shot")
    assert rep.per_query, "no evaluable queries on fixture"
    for q in rep.per_query:
        for k, v in q.recall_at_k.items():
            assert 0.0 <= v <= 1.0
        assert 0.0 <= q.ap <= 1.0
    assert 0.0 <= rep.mAP <= 1.0
    # engineered transitions are perfectly separable for the mock encoder
    perm = {q.text: q for q in rep.per_query}
    soil_q = next(q for q in rep.per_query if "soil" in q.text)
    assert soil_q.recall_at_k[1] == 1.0 and soil_q.ap == 1.0


def test_naive_and_peft_paths_run(store):
    from src.model import create_projection_head
    r = ChangeRetriever(store, MockEncoder())
    s_naive = r.score_all("new water body or flooding", approach="naive")
    assert s_naive.shape == (6,)
    adapter = create_projection_head(input_dim=len(_CLASSES),
                                     output_dim=len(_CLASSES), hidden_dims=(16,))
    r.set_adapter(adapter, feature_mode="difference")
    s_peft = r.score_all("deforestation to soil", approach="peft")
    assert s_peft.shape == (6,) and np.isfinite(s_peft).all()
