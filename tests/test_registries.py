"""
Lock the dataset / encoder / queries registry contract so dataset-extension
branches cannot quietly break the merge invariants documented in
``docs/ARCHITECTURE.md``.
"""
from pathlib import Path

import pytest

from src.datasets.registry import build_dataset, list_datasets
from src.encoders import list_encoders
from src.queries import get_queries

FIXTURE = Path("tests/fixtures/den_tiny")


def test_dataset_registry_has_builtins():
    assert {"dynamic_earthnet", "qfabric_teo"}.issubset(set(list_datasets()))


def test_encoder_registry_has_three():
    assert {"clip_vitl14", "georsclip", "remoteclip"}.issubset(set(list_encoders()))


def test_den_queries_registered_and_nonempty():
    qs = get_queries("dynamic_earthnet")
    assert qs and all(hasattr(q, "predicate") and hasattr(q, "text") for q in qs)


def test_build_dataset_generic_no_dataset_conditional():
    if not FIXTURE.exists():
        pytest.skip("fixture missing")
    ds = build_dataset("dynamic_earthnet", root=str(FIXTURE),
                       pairing="bimonthly", split="test")
    assert ds.name == "dynamic_earthnet"
    assert len(ds.list_pairs()) > 0
