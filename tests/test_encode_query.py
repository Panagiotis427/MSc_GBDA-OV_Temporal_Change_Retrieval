"""
Fast test for ``src.benchmark.encode_query`` — in particular the prompt-ensemble
path (the only multi-template text-scoring path in the pipeline), which was
previously untested. Uses a deterministic mock text encoder (no CLIP/network).
"""
from __future__ import annotations

import numpy as np

from src.benchmark import PROMPT_TEMPLATES, encode_query


class _MockTextEnc:
    name = "mock"
    embed_dim = 8
    image_input_size = 8

    def encode_text(self, texts, batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            # Deterministic across processes (no reliance on hash() randomisation):
            # seed an RNG from the string's codepoints -> a reproducible unit vector.
            seed = (len(t) * 31 + sum(ord(c) for c in t)) % (2**32)
            v = np.random.default_rng(seed).standard_normal(self.embed_dim).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def test_encode_query_single_is_unit_vector():
    enc = _MockTextEnc()
    v = encode_query(enc, "new buildings", ensemble=False)
    assert v.shape == (enc.embed_dim,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_encode_query_ensemble_is_renormalised_and_differs():
    enc = _MockTextEnc()
    single = encode_query(enc, "new buildings", ensemble=False)
    ens = encode_query(enc, "new buildings", ensemble=True)

    assert ens.shape == (enc.embed_dim,)
    # Averaging several template embeddings then renormalising -> still unit norm.
    assert np.isclose(np.linalg.norm(ens), 1.0, atol=1e-5)
    # With >1 distinct template the ensemble must differ from the single-template vector.
    assert len(PROMPT_TEMPLATES) > 1
    assert not np.allclose(ens, single)
