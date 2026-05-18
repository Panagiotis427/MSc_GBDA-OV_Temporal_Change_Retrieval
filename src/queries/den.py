"""
Dynamic EarthNet query set + relevance predicates over DEN's 7 LULC classes.

Self-registers under the dataset name ``"dynamic_earthnet"`` when this module
is imported (``src.queries`` imports it at package load).
"""
from __future__ import annotations

from src.benchmark import Query, _t1, _transition
from . import register_queries

# DEN LULC class names (must match src.datasets.dynamic_earthnet.CLASS_NAMES).
IMPERVIOUS = "impervious_surface"
AGRI = "agriculture"
FOREST = "forest_and_other_vegetation"
WETLANDS = "wetlands"
SOIL = "soil"
WATER = "water"
SNOW = "snow_and_ice"


QUERIES = [
    Query("new buildings constructed on former agricultural land",
          "permanent", _transition(AGRI, IMPERVIOUS)),
    Query("urban expansion replacing vegetation",
          "permanent", lambda lb: _transition(dst=IMPERVIOUS)(lb)
          and _t1(lb) in (AGRI, FOREST)),
    Query("deforestation, forest cleared to bare soil",
          "permanent", _transition(FOREST, SOIL)),
    Query("forest loss",
          "permanent", lambda lb: _t1(lb) == FOREST and not (lb is None or lb.stable)
          and lb.dominant_t2_class not in (FOREST, None)),
    Query("new water body or flooding",
          "permanent", _transition(dst=WATER)),
    Query("bare soil or land cleared",
          "permanent", _transition(dst=SOIL)),
    Query("seasonal snow melting away",
          "seasonal", _transition(src=SNOW)),
    Query("agricultural land converted to wetland or marsh",
          "permanent", _transition(AGRI, WETLANDS)),
    Query("wetland drained and turned into farmland",
          "permanent", _transition(WETLANDS, AGRI)),
    Query("land turning into wetland",
          "permanent", _transition(dst=WETLANDS)),
]


register_queries("dynamic_earthnet", QUERIES)
