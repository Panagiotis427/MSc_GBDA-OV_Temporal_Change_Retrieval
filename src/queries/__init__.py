"""
Per-dataset query-set registry.

A dataset's query set lives in its own module (e.g. ``src/queries/den.py``)
which calls :func:`register_queries`. ``src.benchmark.run_benchmark`` resolves
queries via ``get_queries(dataset.name)``. Adding a dataset's queries = adding
one module here; no edits to shared files.
"""
from __future__ import annotations

from typing import Dict, List

from src.benchmark import Query

_QUERIES: Dict[str, List[Query]] = {}


def register_queries(name: str, queries: List[Query]) -> None:
    _QUERIES[name] = list(queries)


def get_queries(name: str) -> List[Query]:
    return list(_QUERIES.get(name, []))


def list_query_sets() -> List[str]:
    return sorted(_QUERIES)


def has_tag(tag: str):
    """Relevance predicate: the label's pipe-joined ``change_type`` contains *tag*.

    Shared by the caption-derived query sets (``levir_cc``, ``second_cc``) whose
    labels store change tags as a ``|``-joined string. ``None`` labels are treated
    as non-relevant.
    """
    return lambda lb: lb is not None and tag in (lb.change_type or "").split("|")


# Auto-register the built-in dataset query sets on first import.
from . import den  # noqa: F401,E402
from . import qfabric  # noqa: F401,E402
from . import qfabric_status  # noqa: F401,E402
from . import levir_cc  # noqa: F401,E402
from . import levir_mci  # noqa: F401,E402
from . import second_cc  # noqa: F401,E402
