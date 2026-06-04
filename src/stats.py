"""Shared statistical helpers for the retrieval-evaluation scripts.

``rand_ap`` is the average precision of a *uniformly random* ranking of ``N``
items with ``R`` relevant — the honest chance baseline for AP (whose expectation
is ~prevalence, not zero). ``scripts/patch_eval`` and ``scripts/significance_audit``
each carried a byte-identical copy of this; this module is the single source of
truth so the permutation baseline can never drift between them.

NOTE: ``scripts/cv_eval`` deliberately keeps its own permutation routine — it
draws via ``rng.permutation`` rather than ``rng.shuffle``, so swapping it for
this helper would change its RNG draw sequence and perturb the already-committed
``cv_eval__*.json`` results. It is intentionally left separate.
"""
from __future__ import annotations

import numpy as np


def rand_ap(R: int, N: int, rng) -> float:
    """AP of a uniformly random ranking of ``N`` items with ``R`` relevant.

    ``rng`` is a ``numpy`` Generator/RandomState; one ``rng.shuffle`` call is
    consumed per invocation (callers rely on this for reproducible draws).
    """
    rel = np.zeros(N, dtype=bool)
    rel[:R] = True
    rng.shuffle(rel)
    hits = np.cumsum(rel)
    return float((hits / np.arange(1, N + 1))[rel].sum() / R)
