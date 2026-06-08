"""
LEVIR-CC open-vocabulary change queries + relevance from caption-derived tags.

LEVIR-CC pairs carry change tags parsed from their five human captions
(``src/datasets/levir_cc.py``); a pair is relevant to a query iff the matching
tag is present in its ``PairLabel.change_type`` (a pipe-joined tag string). All
queries are permanent --- LEVIR-CC has no seasonal class.

Self-registers on import (``src.queries`` imports this at package load).
"""
from __future__ import annotations

from src.benchmark import Query
from . import register_queries


def _has(tag: str):
    return lambda lb: lb is not None and tag in (lb.change_type or "").split("|")


QUERIES = [
    Query("new buildings or houses constructed", "permanent", _has("building")),
    Query("a new road or street built", "permanent", _has("road")),
    Query("buildings demolished or removed", "permanent", _has("demolition")),
]

register_queries("levir_cc", QUERIES)
