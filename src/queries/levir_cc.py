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
from . import has_tag, register_queries


# The loader (src/datasets/levir_cc.py) parses five change tags from the human
# captions; the three construction tags plus the two land-cover tags below. The
# vegetation/water queries exercise tags the loader always produced but no query
# previously tested. Test-split positives: vegetation 462 (strong), water 19
# (sparse, prevalence ~0.01 — a deliberately weak open-vocab probe, reported as
# such, never a headline).
QUERIES = [
    Query("new buildings or houses constructed", "permanent", has_tag("building")),
    Query("a new road or street built", "permanent", has_tag("road")),
    Query("buildings demolished or removed", "permanent", has_tag("demolition")),
    Query("trees or vegetation cleared or grown", "permanent", has_tag("vegetation")),
    Query("a lake, pond, or body of water", "permanent", has_tag("water")),
]

register_queries("levir_cc", QUERIES)
