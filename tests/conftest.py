"""
Ensure the deterministic synthetic DEN fixture exists before tests run, so the
fast suite is self-contained (the fixture is gitignored — it is generated, not
checked in).
"""
from pathlib import Path

from scripts.make_den_fixture import build_fixture

_FIXTURE = Path("tests/fixtures/den_tiny")


def pytest_configure(config):
    if not (_FIXTURE / "_done.marker").exists():
        build_fixture(_FIXTURE, force=True)
