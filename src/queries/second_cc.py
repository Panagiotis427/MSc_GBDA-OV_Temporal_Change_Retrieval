"""
SECOND-CC open-vocabulary change queries + relevance from caption-derived tags.

SECOND-CC pairs carry change tags parsed from their human captions
(``src/datasets/second_cc.py``); a pair is relevant to a query iff the matching
tag is present in its ``PairLabel.change_type``. The query set spans SECOND's six
semantic classes plus road (mentioned heavily in captions though it has no
semantic-map class) -- the open-vocabulary *breadth* LEVIR-CC lacks. All queries
are permanent (SECOND-CC has no seasonal class).

The six class-involved query texts match the keys of
``second_cc.QUERY_TO_MASK_CLASS`` so ``scripts/eval_localization.py`` can score
each against its semantic-map change mask; the road query has no mask (reported
honestly).

Self-registers on import (``src.queries`` imports this at package load).
"""
from __future__ import annotations

from src.benchmark import Query
from . import register_queries


def _has(tag: str):
    return lambda lb: lb is not None and tag in (lb.change_type or "").split("|")


QUERIES = [
    Query("new buildings or structures appeared", "permanent", _has("building")),
    Query("a new road or street", "permanent", _has("road")),
    Query("trees appeared or were cleared", "permanent", _has("tree")),
    Query("low vegetation or grassland changed", "permanent", _has("low_vegetation")),
    Query("a water body appeared or changed", "permanent", _has("water")),
    Query("bare ground or land cleared", "permanent", _has("ground")),
    Query("a playground or sports field", "permanent", _has("playground")),
]

register_queries("second_cc", QUERIES)
