"""
QFabric *status-transition* query set + relevance predicates (RQA5).

Targets the ``qfabric_status`` dataset, whose ``PairLabel`` carries the
per-timepoint development statuses as ``dominant_t1_class`` (status@t1) and
``dominant_t2_class`` (status@t2). A pair is relevant to a query iff its
transition matches ``src -> dst`` — so **stable pairs (status unchanged) are
non-relevant negatives**, which is exactly what isolates the directional
``zero_shot`` Δ-signal from the after-image content ``naive`` keys on.

All queries are ``permanent`` (QFabric has no seasonal class; seasonal-drift@K
is reported as 0, correctly). Self-registers on import.
"""
from __future__ import annotations

from typing import Iterable

from src.benchmark import Query

from . import register_queries


def _trans(src: Iterable[str], dst: Iterable[str]):
    """Relevant iff status@t1 in *src* and status@t2 in *dst* (a real transition)."""
    src_set, dst_set = set(src), set(dst)
    return lambda lb: (lb is not None
                       and lb.dominant_t1_class in src_set
                       and lb.dominant_t2_class in dst_set)


_PRE = ("greenland", "prior_construction")                       # not-yet-built
_ACTIVE = ("land_cleared", "excavation", "materials_dumped",
           "construction_started", "construction_midway")        # in-progress
_DONE = ("construction_done", "operational")                     # finished


QUERIES = [
    Query("land cleared and prepared for new development", "permanent",
          _trans(_PRE, ("land_cleared",))),
    Query("excavation and earthworks begun on the site", "permanent",
          _trans((*_PRE, "land_cleared"), ("excavation", "materials_dumped"))),
    Query("new building construction has started", "permanent",
          _trans((*_PRE, "land_cleared", "excavation", "materials_dumped"),
                 ("construction_started", "construction_midway"))),
    Query("building construction recently completed", "permanent",
          _trans(("land_cleared", "excavation", "materials_dumped",
                  "construction_started", "construction_midway"), _DONE)),
    Query("buildings demolished, site cleared back to bare land", "permanent",
          _trans((*_DONE, "construction_started", "construction_midway"),
                 ("land_cleared", "greenland"))),
    Query("vacant land developed into a finished building", "permanent",
          _trans(_PRE, _DONE)),
]

register_queries("qfabric_status", QUERIES)
