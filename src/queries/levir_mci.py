"""
LEVIR-MCI retrieval query set --- identical to LEVIR-CC.

LEVIR-MCI shares LEVIR-CC's captions and tags, so the open-vocabulary retrieval
queries are the same five (building, road, demolition, vegetation, water). MCI's
addition is pixel-level masks for building/road change, consumed by
``scripts/eval_localization.py`` for localization metrics, not retrieval.

Self-registers on import (``src.queries`` imports this at package load).
"""
from __future__ import annotations

from . import register_queries
from .levir_cc import QUERIES

register_queries("levir_mci", QUERIES)
