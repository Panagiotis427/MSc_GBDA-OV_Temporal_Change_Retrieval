"""Guards the BH-FDR step-up in ``scripts.significance_audit.collect``.

The ``bh_fdr`` column it computes feeds the committed ``results_audit_summary.csv``
and the q-values cited in the report, so a regression in the step-up would
silently corrupt a statistical claim. These tests pin the textbook properties:
monotone q-values, capped at 1.0, ``m==1`` boundary, and train cells excluded
from the FDR family.
"""
from __future__ import annotations

import json

from scripts.significance_audit import collect


def _write_result(path, *, split, n_pairs, n_rel, ap, approach="zero_shot",
                  lora=False):
    path.write_text(json.dumps({
        "dataset": "den", "encoder": "georsclip", "split": split,
        "color_mode": "rgb", "approach": approach, "lora": lora,
        "n_pairs": n_pairs, "macro": {"mAP": ap},
        "per_query": [{"n_relevant": n_rel, "ap": ap}],
    }))


def test_bh_fdr_monotone_capped_and_train_excluded(tmp_path):
    # Three held-out cells spanning strong -> chance, plus one train (leakage) cell.
    _write_result(tmp_path / "a.json", split="test", n_pairs=110, n_rel=3, ap=0.60)
    _write_result(tmp_path / "b.json", split="val", n_pairs=110, n_rel=3, ap=0.90)
    _write_result(tmp_path / "c.json", split="test", n_pairs=110, n_rel=3, ap=0.99)
    _write_result(tmp_path / "d.json", split="train", n_pairs=600, n_rel=5, ap=0.40,
                  approach="peft")

    rows = collect(str(tmp_path))
    held = sorted((r for r in rows if r["split"] in ("test", "eval", "val")),
                  key=lambda r: r["perm_p"])

    # q-values are monotone non-decreasing in p-rank and capped to [0, 1].
    qs = [r["bh_fdr"] for r in held]
    assert qs == sorted(qs), f"bh_fdr not monotone in p-rank: {qs}"
    assert all(0.0 <= q <= 1.0 for q in qs)

    # Train cells are outside the FDR family -> no q-value assigned.
    train = [r for r in rows if r["split"] == "train"]
    assert train and all(r["bh_fdr"] == "" for r in train)


def test_bh_fdr_single_held_out_cell(tmp_path):
    # m == 1: the lone held-out cell's q-value is just its own p (m/rank == 1).
    _write_result(tmp_path / "only.json", split="test", n_pairs=110, n_rel=3, ap=0.50)
    rows = collect(str(tmp_path))
    held = [r for r in rows if r["split"] == "test"]
    assert len(held) == 1
    r = held[0]
    assert r["bh_fdr"] == round(min(1.0, r["perm_p"]), 4)
