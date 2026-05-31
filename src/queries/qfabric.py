"""
QFabric query set + relevance predicates over the 6 change types.

Targets the TEOChatlas-QFabric dataset (``qfabric_teo``), whose ``PairLabel``
carries the real change type. A pair is relevant to a query iff its crop's
change type matches. All queries are ``permanent`` (QFabric has no seasonal
class — seasonal-drift@K is reported as 0, correctly).

Self-registers on import (``src.queries`` imports this at package load).
"""
from __future__ import annotations

from src.benchmark import Query
from . import register_queries


def _is(change_type: str):
    return lambda lb: lb is not None and lb.change_type == change_type


QUERIES = [
    Query("new residential housing construction", "permanent", _is("residential")),
    Query("new commercial buildings or retail development", "permanent", _is("commercial")),
    Query("new industrial facility or factory construction", "permanent", _is("industrial")),
    Query("new road or highway construction", "permanent", _is("road")),
    Query("buildings demolished, structures torn down", "permanent", _is("demolition")),
    Query("large-scale mega project under construction", "permanent", _is("mega_projects")),
]

register_queries("qfabric_teo", QUERIES)
