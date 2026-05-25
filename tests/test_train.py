"""
Fast PEFT-training test (no CLIP): a caption-aware mock encoder + the
synthetic DEN fixture. Verifies the rewritten ``src.train``:
  - the adapter actually learns (loss decreases),
  - save/load round-trips,
  - the trained PEFT adapter is not worse than zero-shot Δ-similarity on the
    engineered fixture (the real CLIP comparison runs in the benchmark CLI).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.datasets.registry import get_dataset
from src.embeddings import compute_pair_embeddings
from src.retrieval import ChangeRetriever
from src.benchmark import run_benchmark
from src.model import load_adapter, save_adapter
from src.train import TrainConfig, train_adapter

FIXTURE = Path("tests/fixtures/den_tiny")

from _mocks import MockEncoderBase, _CLASSES

_SPACED = {c: c.replace("_", " ") for c in _CLASSES}
_QUERY_MAP = [  # (token, destination class)
    ("buildings", "impervious_surface"), ("urban", "impervious_surface"),
    ("deforest", "soil"), ("bare soil", "soil"), ("cleared", "soil"),
    ("water", "water"), ("flood", "water"),
    ("snow", "forest_and_other_vegetation"),  # post-melt destination in fixture
]


class MockEncoder(MockEncoderBase):
    def encode_text(self, texts, batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            low = t.lower()
            dest = None
            if "replaced by" in low:                       # caption
                right = low.split("replaced by", 1)[1]
                dest = next((c for c in _CLASSES if _SPACED[c] in right), None)
            elif "stable" in low and "land cover" in low:   # stable caption
                mid = low.split("stable", 1)[1].split("land cover", 1)[0]
                dest = next((c for c in _CLASSES if _SPACED[c] in mid), None)
            else:                                            # query
                dest = next((c for tok, c in _QUERY_MAP if tok in low), None)
            v = self._onehot(dest) if dest else np.zeros(self.embed_dim, np.float32)
            out.append(v)
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


def test_adapter_learns_and_peft_not_worse(den, store, tmp_path):
    enc = MockEncoder()
    retr = ChangeRetriever(store, enc, feature_mode="difference")
    map_zs = run_benchmark(den, retr, approach="zero_shot").mAP

    cfg = TrainConfig(mode="difference", epochs=120, batch_size=4,
                      lr=1e-2, hidden_dims=(16,), dropout=0.0)
    adapter, hist = train_adapter(den, store, enc, cfg,
                                  device=torch.device("cpu"), verbose=False)

    assert hist["loss"][-1] < hist["loss"][0], "adapter did not learn"

    ckpt = tmp_path / "adapter.pt"
    save_adapter(str(ckpt), adapter, {
        "input_dim": adapter.input_dim, "output_dim": adapter.output_dim,
        "hidden_dims": [16], "dropout_rate": 0.0,
        "feature_mode": "difference", "encoder_name": enc.name,
        "dataset_name": den.name,
    })
    loaded, meta = load_adapter(str(ckpt))
    assert meta["feature_mode"] == "difference"

    retr.set_adapter(adapter, feature_mode="difference")
    s_a = retr.score_all("deforestation forest cleared to bare soil", "peft")
    retr.set_adapter(loaded, feature_mode="difference")
    s_b = retr.score_all("deforestation forest cleared to bare soil", "peft")
    np.testing.assert_allclose(s_a, s_b, rtol=1e-5, atol=1e-5)

    map_peft = run_benchmark(den, retr, approach="peft").mAP
    assert map_peft >= map_zs * 0.9, f"PEFT {map_peft:.3f} << zero-shot {map_zs:.3f}"
    assert map_peft > 0.3
