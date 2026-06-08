"""Tests for the shared evaluation-statistics helpers in ``src.stats``.

These cover the machinery behind the report's audited numbers: the unbiased
permutation p-value, deterministic pessimistic ranking, Benjamini-Hochberg FDR,
and the leakage-free k-fold AOI partition shared by cv_eval / patch_eval.
"""
import numpy as np
import pytest

from src.stats import aoi_folds, bh_fdr, perm_p_value, rand_ap, rank_order


class TestPermPValue:
    def test_never_zero(self):
        # Even with zero null draws >= observed, p = 1/(iters+1) > 0.
        assert perm_p_value(0, 4000) == pytest.approx(1 / 4001)
        assert perm_p_value(0, 4000) > 0.0

    def test_all_exceed_gives_one(self):
        assert perm_p_value(4000, 4000) == pytest.approx(1.0)

    def test_monotone_in_count(self):
        assert perm_p_value(10, 4000) < perm_p_value(100, 4000)


class TestRankOrder:
    def test_pessimistic_tie_break(self):
        # Tie at the top score between a relevant (idx 0) and non-relevant (idx 1):
        # the non-relevant item must rank first (pessimistic).
        scores = np.array([0.5, 0.5, 0.1])
        rel = np.array([True, False, True])
        order = rank_order(scores, rel)
        assert list(order) == [1, 0, 2]

    def test_deterministic(self):
        scores = np.array([0.3, 0.3, 0.3, 0.3])
        rel = np.array([True, True, False, False])
        a = rank_order(scores, rel)
        b = rank_order(scores, rel)
        assert list(a) == list(b)

    def test_pessimistic_never_above_optimistic_ap(self):
        # On a full tie, pessimistic AP equals prevalence (relevant last).
        scores = np.zeros(4)
        rel = np.array([True, False, False, False])
        order = rank_order(scores, rel)
        # the single relevant item is placed last among the tie
        assert order[-1] == 0


class TestBHFDR:
    def test_matches_manual(self):
        q = bh_fdr([0.01, 0.02, 0.03])
        # m=3: raw p*m/rank = [0.03, 0.03, 0.03] -> all 0.03 after monotone step-up
        assert np.allclose(q, [0.03, 0.03, 0.03])

    def test_capped_at_one_and_order_preserved(self):
        q = bh_fdr([0.9, 0.001, 0.5])
        assert q.max() <= 1.0
        # the smallest p keeps the smallest q
        assert np.argmin(q) == 1

    def test_empty(self):
        assert bh_fdr([]).size == 0


class TestAOIFolds:
    AOIS = [f"aoi_{i}" for i in range(75)]

    def test_disjoint_and_complete(self):
        folds = aoi_folds(self.AOIS, 5, seed=42)
        assert set(folds.values()) == {0, 1, 2, 3, 4}
        # every AOI assigned exactly once (dict keys) and all covered
        assert set(folds.keys()) == set(self.AOIS)
        # roughly balanced
        sizes = [sum(1 for v in folds.values() if v == k) for k in range(5)]
        assert max(sizes) - min(sizes) <= 1

    def test_identical_for_same_seed(self):
        # This is the guarantee that cv_eval and patch_eval share one partition.
        a = aoi_folds(self.AOIS, 5, seed=42)
        b = aoi_folds(self.AOIS, 5, seed=42)
        assert a == b

    def test_order_independent(self):
        # Shuffled input AOIs -> same partition (helper sorts internally).
        shuffled = list(reversed(self.AOIS))
        assert aoi_folds(self.AOIS, 5, 42) == aoi_folds(shuffled, 5, 42)

    def test_no_aoi_spans_two_folds(self):
        folds = aoi_folds(self.AOIS, 5, seed=7)
        groups = {}
        for aoi, k in folds.items():
            groups.setdefault(k, set()).add(aoi)
        all_assigned = [a for g in groups.values() for a in g]
        assert len(all_assigned) == len(set(all_assigned)) == 75


def test_rand_ap_near_prevalence():
    rng = np.random.default_rng(0)
    R, N = 8, 100
    draws = [rand_ap(R, N, rng) for _ in range(500)]
    # expected random AP ~ prevalence R/N, with a small upward bias
    assert R / N - 0.02 < np.mean(draws) < R / N + 0.05
