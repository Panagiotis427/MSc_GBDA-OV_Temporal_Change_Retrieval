"""
Ensure the deterministic synthetic DEN fixture exists before tests run, so the
fast suite is self-contained. The fixture is committed to the repo; this only
regenerates it if it is missing (e.g. a contributor deleted it).
"""
from pathlib import Path

from scripts.make_den_fixture import build_fixture

_FIXTURE = Path("tests/fixtures/den_tiny")


def pytest_configure(config):
    if not (_FIXTURE / "_done.marker").exists():
        build_fixture(_FIXTURE, force=True)
