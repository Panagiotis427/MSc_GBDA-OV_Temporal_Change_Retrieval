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
    assert "File" in kinds, "CSV download (gr.File) missing"
    # Three per-image download buttons: Before / After / heatmap.
    assert kinds.count("DownloadButton") >= 3, "expected before/after/heatmap download buttons"
    # All-matches grid: MAX_RESULTS (10) image tiles, each with its own View button.
    assert kinds.count("Image") >= 10, "expected 10 all-matches tile images"
    assert kinds.count("Button") >= 10, "expected per-tile View buttons"


def test_results_csv_clean_name_and_content(tmp_path):
    """The CSV export has a usable, dataset-named basename ending in .csv and
    contains the header + every row (the download bug was a random extension-less
    temp name)."""
    import csv as _csv
    from src.app import results_to_csv, _CSV_HEADER

    assert results_to_csv([], "levir_mci") is None  # no rows -> no file
    rows = [[1, "loc_A", "2018-05", "2018-07", 0.42, 0.91, "new buildings", "permanent"],
            [2, "loc_B", "2018-05", "2018-07", 0.31, 0.55, "new road", "permanent"]]
    path = results_to_csv(rows, "levir_mci")
    assert path is not None
    base = Path(path).name
    assert base == "change_results_levir_mci.csv", base
    with open(path, newline="", encoding="utf-8") as fh:
        read = list(_csv.reader(fh))
    assert read[0] == _CSV_HEADER
    assert len(read) == 1 + len(rows)
    assert read[1][1] == "loc_A"


def test_materialize_image_named_png():
    """Result images are saved with meaningful basenames so downloads aren't all
    called 'image.png'."""
    from PIL import Image as _Im
    from src.app import materialize_image
    assert materialize_image(None, "x") is None
    p = materialize_image(_Im.new("RGB", (8, 8), "red"), "after_2065_2018-07-01")
    assert Path(p).name == "after_2065_2018-07-01.png", Path(p).name
    p2 = materialize_image(_Im.new("RGB", (8, 8), "red"), "weird/name?")
    assert Path(p2).name == "weird_name_.png", Path(p2).name


def test_results_csv_sanitizes_dataset_name():
    """A dataset key with path-unsafe characters still yields a safe basename."""
    from src.app import results_to_csv
    path = results_to_csv([[1, "x", "a", "b", 0.1, 0.2, "c", "d"]], "weird/name :*?")
    assert Path(path).name == "change_results_weird_name____.csv", Path(path).name
