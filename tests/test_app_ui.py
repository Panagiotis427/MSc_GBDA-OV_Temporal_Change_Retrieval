"""Headless construction test for the Gradio UI wiring (no browser, no launch).

Building the Blocks tree validates that every event handler's declared inputs /
outputs match the components actually created — an arity mismatch (e.g. after
adding or removing an output) raises here, not in front of a user. Also asserts
the before/after swipe slider and the CSV-download control are wired in.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.datasets.registry import get_dataset
from src.embeddings import compute_pair_embeddings
from src.app import RunConfig, SemanticChangeSearch

FIXTURE = Path("tests/fixtures/den_tiny")
from _mocks import MockEncoderBase  # noqa: E402  (tests/ on path via conftest)


class MockEncoder(MockEncoderBase):
    def encode_text(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        return np.stack([np.zeros(self.embed_dim, np.float32) for _ in texts])


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


def test_interface_constructs(engine):
    """build_interface() wires every click/submit/load handler without an
    input/output arity mismatch."""
    import gradio as gr
    demo = engine.build_interface()
    assert isinstance(demo, gr.Blocks)


def test_swipe_and_download_present(engine):
    demo = engine.build_interface()
    kinds = [type(b).__name__ for b in demo.blocks.values()]
    # Two swipe sliders: Before↔After and After↔heatmap.
    assert kinds.count("ImageSlider") >= 2, "expected before/after + after/heatmap sliders"
    assert "DownloadButton" in kinds, "CSV download control missing"
    assert "Gallery" in kinds, "top-K gallery missing"
