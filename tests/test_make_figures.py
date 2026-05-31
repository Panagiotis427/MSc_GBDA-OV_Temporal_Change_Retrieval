"""Smoke tests for scripts/make_figures.py — synthetic schema-shaped records,
assert each figure writes a non-empty PNG. No torch, no real results needed.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from scripts import make_figures as mf

_KS = [1, 3, 5, 10]


def _rec(encoder, approach, split, color="rgb", lora=False, mAP=0.3):
    return {
        "dataset": "dynamic_earthnet", "encoder": encoder, "approach": approach,
        "split": split, "color_mode": color, "lora": lora,
        "n_pairs": 605, "k_values": _KS,
        "macro": {
            "mAP": mAP,
            "recall_at_k": {str(k): mAP * (k / 10.0) for k in _KS},
            "seasonal_drift_at_k": {str(k): 0.1 for k in _KS},
        },
        "per_query": [],
    }


def _records():
    recs = []
    for enc in ("clip_vitl14", "georsclip", "remoteclip"):
        for sp in ("train", "val", "test"):
            for ap in ("naive", "zero_shot", "peft"):
                recs.append(_rec(enc, ap, sp, mAP=0.4))
            for color in ("nrg", "ndvi"):
                recs.append(_rec(enc, "zero_shot", sp, color=color, mAP=0.25))
    # a lora record (georsclip nrg)
    recs.append(_rec("georsclip", "zero_shot", "test", color="nrg", lora=True, mAP=0.16))
    return recs


def test_recall_curves(tmp_path):
    p = mf.fig_recall_curves(_records(), tmp_path, split="train", color="rgb")
    assert p is not None and p.exists() and p.stat().st_size > 0


def test_map_bars(tmp_path):
    p = mf.fig_map_grouped_bars(_records(), tmp_path, split="train", color="rgb")
    assert p is not None and p.exists() and p.stat().st_size > 0


def test_color_ablation(tmp_path):
    p = mf.fig_color_ablation_heatmap(_records(), tmp_path, approach="zero_shot", split="test")
    assert p is not None and p.exists() and p.stat().st_size > 0


def test_seasonal_drift(tmp_path):
    p = mf.fig_seasonal_drift(_records(), tmp_path, split="train", color="rgb")
    assert p is not None and p.exists() and p.stat().st_size > 0


def test_cross_split(tmp_path):
    p = mf.fig_cross_split_map(_records(), tmp_path, encoder="clip_vitl14", color="rgb")
    assert p is not None and p.exists() and p.stat().st_size > 0


def test_graceful_skip_missing(tmp_path):
    # No records for this color -> returns None, writes nothing.
    assert mf.fig_recall_curves(_records(), tmp_path, split="train", color="zzz") is None
    assert not list(tmp_path.glob("*.png"))
