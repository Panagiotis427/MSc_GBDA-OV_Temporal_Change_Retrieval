"""Unit tests for the pure-numpy localization metric helpers in
``scripts/eval_localization.py`` (no GPU / no dataset needed)."""
import numpy as np

from scripts.eval_localization import _average_precision, _gt_patches


def test_average_precision_perfect_ranking():
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    labels = np.array([1, 0, 1, 0], dtype=bool)
    # ranked: 0.9(pos), 0.8(pos), 0.2(neg), 0.1(neg) -> precision 1.0 at both hits
    assert _average_precision(scores, labels) == 1.0


def test_average_precision_worst_ranking():
    scores = np.array([0.9, 0.1])
    labels = np.array([0, 1], dtype=bool)   # the one positive is ranked last
    assert _average_precision(scores, labels) == 0.5


def test_average_precision_no_positives_is_zero():
    assert _average_precision(np.array([0.9, 0.1]), np.array([0, 0], dtype=bool)) == 0.0


def test_gt_patches_downsamples_block_to_one_cell():
    mask = np.zeros((8, 8), dtype=bool)
    mask[0:4, 0:4] = True                    # top-left quadrant changed
    gt = _gt_patches(mask, side=2, pos_thresh=0.0)
    assert gt.shape == (4,)                  # 2x2 grid flattened
    assert gt.tolist() == [True, False, False, False]


def test_gt_patches_threshold_filters_sparse_cells():
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, 0] = True                        # 1 px in the top-left 4x4 cell -> fraction 1/16
    # any-pixel threshold keeps it; a 0.1 threshold (needs >10% of the cell) drops it
    assert _gt_patches(mask, side=2, pos_thresh=0.0)[0]
    assert not _gt_patches(mask, side=2, pos_thresh=0.1)[0]


def test_gt_patches_empty_mask_all_false():
    assert not _gt_patches(np.zeros((8, 8), dtype=bool), side=2, pos_thresh=0.0).any()
