"""Tests for src/error_analysis.py — _actual_transition unit cases + a
deterministic confusion build on the synthetic DEN fixture (mock encoder).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.datasets.base import PairLabel
from src.datasets.registry import get_dataset
from src.embeddings import compute_pair_embeddings
from src.error_analysis import (ConfusionReport, _actual_transition,
                                build_confusion)
from src.retrieval import ChangeRetriever

FIXTURE = Path("tests/fixtures/den_tiny")

from _mocks import _CLASSES  # noqa: E402
from test_change_retrieval import MockEncoder  # reuse the keyword->class mock


def _lb(t1, t2, stable=False):
    return PairLabel(change_type=f"{t1}->{t2}", stable=stable,
                     dominant_t1_class=t1, dominant_t2_class=t2)


def test_actual_transition_cases():
    assert _actual_transition(None) == "unlabeled"
    assert _actual_transition(_lb("soil", "soil", stable=True)) == "stable"
    assert _actual_transition(_lb("forest_and_other_vegetation", "soil")) \
        == "forest_and_other_vegetation->soil"
    # snow present -> seasonal column
    assert _actual_transition(_lb("snow_and_ice", "soil")).startswith("seasonal:")
    assert _actual_transition(_lb("forest_and_other_vegetation",
                                  "forest_and_other_vegetation")) == "stable"


@pytest.fixture(scope="module")
def den():
    if not FIXTURE.exists():
        pytest.skip("DEN fixture missing; run scripts.make_den_fixture")
    return get_dataset("dynamic_earthnet", root=str(FIXTURE),
                       pairing_strategy="bimonthly")


@pytest.fixture(scope="module")
def store(den):
    return compute_pair_embeddings(den, MockEncoder())


def test_build_confusion_shape_and_precision(den, store):
    r = ChangeRetriever(store, MockEncoder())
    report = build_confusion(den, r, approach="zero_shot", split="train")
    assert report.matrix.shape == (len(report.per_query), len(report.labels))
    assert report.per_query, "no evaluable queries on fixture"
    # the deforestation->soil query is perfectly separable for the mock encoder
    soil_q = next(q for q in report.per_query if "soil" in q.text)
    assert soil_q.precision_at_k[1] == 1.0
    # precision/recall are valid probabilities
    for q in report.per_query:
        for v in list(q.precision_at_k.values()) + list(q.recall_at_k.values()):
            assert 0.0 <= v <= 1.0


def test_confusion_report_round_trip(den, store):
    r = ChangeRetriever(store, MockEncoder())
    report = build_confusion(den, r, approach="zero_shot", split="test")
    back = ConfusionReport.from_dict(report.to_dict())
    assert back.labels == report.labels
    assert back.encoder == report.encoder
    np.testing.assert_array_equal(back.matrix, report.matrix)
    assert [q.text for q in back.per_query] == [q.text for q in report.per_query]
