"""
Cross-dataset open-vocabulary retrieval figures from ``results/*.json``.

Pure consumer of the per-dataset benchmark JSON (written by
``scripts.benchmark_levir_cc`` / ``benchmark_second_cc`` / ``benchmark_qfabric``).
Imports only ``json`` + ``matplotlib`` + ``numpy`` --- no torch, no models, no
network --- so it is safe to run anywhere the result JSON exists.

Figures (into ``report/figures/``)
----------------------------------
- ``per_query_ap__levir_cc__test.png``   per-query zero-shot AP, 3 encoders + floor
- ``per_query_ap__second_cc__test.png``  per-query zero-shot AP, 3 encoders + floor
- ``qfabric_status_map__eval.png``       status-transition mAP, naive vs zero-shot
- ``salience_law__summary.png``          cross-dataset AP vs prevalence (the law)

Run::

    python -m scripts.make_dataset_figures --results-dir results --out-dir report/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts._figutil import save_fig as _save  # noqa: E402

_ENCODERS = ["georsclip", "clip_vitl14", "remoteclip"]
_ENC_LABEL = {"georsclip": "GeoRSCLIP", "clip_vitl14": "CLIP ViT-L/14",
              "remoteclip": "RemoteCLIP"}


def _load(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _short(text: str, n: int = 22) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


# ----------------------------------------------------------------------
# per-query AP bars (LEVIR-CC / SECOND-CC caption schema)
# ----------------------------------------------------------------------
def fig_per_query_ap(results_dir: Path, out_dir: Path, *, dataset: str,
                     split: str = "test", approach: str = "zero_shot",
                     svg: bool = False) -> Optional[Path]:
    data = {enc: _load(results_dir / f"{dataset}__{enc}__{split}.json")
            for enc in _ENCODERS}
    data = {k: v for k, v in data.items() if v is not None}
    if not data:
        print(f"  skip per_query_ap__{dataset}__{split}: no records")
        return None
    ref = next(iter(data.values()))
    queries = list(ref["query_prevalence"].keys())
    floors = [ref["query_prevalence"][q] for q in queries]
    encs = [e for e in _ENCODERS if e in data]

    x = np.arange(len(queries))
    w = 0.8 / max(len(encs), 1)
    fig, ax = plt.subplots(figsize=(1.4 * len(queries) + 3, 5))
    for i, enc in enumerate(encs):
        ap = data[enc]["approaches"][approach]["per_query_ap"]
        vals = [ap.get(q, 0.0) for q in queries]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=_ENC_LABEL[enc])
    # prevalence (random-AP) floor as a step marker per query
    for xi, fl in zip(x, floors):
        ax.hlines(fl, xi - 0.42, xi + 0.42, colors="black",
                  linestyles="dashed", linewidth=1.2,
                  label="random-AP floor (prevalence)" if xi == 0 else None)
    ax.set_xticks(x)
    n_rel = ref["approaches"][approach].get("per_query_n_relevant", {})
    ax.set_xticklabels([f"{_short(q)}\n(n={n_rel.get(q,'?')})" for q in queries],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"average precision ({approach.replace('_', '-')})")
    ax.set_title(f"{dataset.upper().replace('_', '-')} per-query AP "
                 f"(split={split}); dashed = prevalence floor")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    return _save(fig, out_dir, f"per_query_ap__{dataset}__{split}", svg)


# ----------------------------------------------------------------------
# QFabric status-transition mAP (per_query list schema, naive vs zero_shot)
# ----------------------------------------------------------------------
def fig_qfabric_status_map(results_dir: Path, out_dir: Path, *, split: str = "eval",
                           color: str = "rgb", svg: bool = False) -> Optional[Path]:
    encs, naive, zshot, floor = [], [], [], None
    for enc in _ENCODERS:
        n = _load(results_dir / f"qfabric_status__{enc}__{split}__{color}__naive.json")
        z = _load(results_dir / f"qfabric_status__{enc}__{split}__{color}__zero_shot.json")
        if n is None or z is None:
            continue
        encs.append(_ENC_LABEL[enc])
        naive.append(n["macro"]["mAP"])
        zshot.append(z["macro"]["mAP"])
        if floor is None:  # macro prevalence = mean per-query positive fraction
            npairs = z.get("n_pairs", 0) or 1
            pq = z.get("per_query", [])
            if pq:
                floor = float(np.mean([q.get("n_relevant", 0) / npairs for q in pq]))
    if not encs:
        print("  skip qfabric_status_map: no records")
        return None
    x = np.arange(len(encs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x - w / 2, naive, w, label="naive")
    ax.bar(x + w / 2, zshot, w, label="zero-shot")
    if floor is not None:
        ax.axhline(floor, color="black", linestyle="dashed", linewidth=1.1,
                   label=f"macro-prevalence floor ({floor:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(encs, rotation=10, ha="right")
    ax.set_ylabel("mAP")
    ax.set_title("QFabric status-transition retrieval (RQA5, RGB)")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
    return _save(fig, out_dir, f"qfabric_status_map__{split}", svg)


# ----------------------------------------------------------------------
# cross-dataset salience law: zero-shot AP vs query prevalence
# ----------------------------------------------------------------------
def _collect_caption(results_dir: Path, dataset: str, encoder: str = "georsclip",
                     split: str = "test"):
    d = _load(results_dir / f"{dataset}__{encoder}__{split}.json")
    if d is None:
        return []
    ap = d["approaches"]["zero_shot"]["per_query_ap"]
    prev = d["query_prevalence"]
    return [(prev[q], ap[q], q) for q in ap if q in prev]


def _collect_qfabric(results_dir: Path, dataset: str, encoder: str = "georsclip",
                     split: str = "eval", color: str = "rgb"):
    d = _load(results_dir / f"{dataset}__{encoder}__{split}__{color}__zero_shot.json")
    if d is None:
        return []
    n_pairs = d.get("n_pairs", 0) or 1
    out = []
    for q in d.get("per_query", []):
        prev = q.get("n_relevant", 0) / n_pairs
        out.append((prev, q["ap"], q["text"]))
    return out


def fig_salience_law(results_dir: Path, out_dir: Path, *, svg: bool = False) -> Optional[Path]:
    series = {
        "LEVIR-CC": ("o", _collect_caption(results_dir, "levir_cc")),
        "SECOND-CC": ("s", _collect_caption(results_dir, "second_cc")),
        "QFabric (type)": ("^", _collect_qfabric(results_dir, "qfabric_teo")),
        "QFabric (status)": ("D", _collect_qfabric(results_dir, "qfabric_status")),
    }
    series = {k: v for k, v in series.items() if v[1]}
    if not series:
        print("  skip salience_law: no records")
        return None
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label, (marker, pts) in series.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, marker=marker, s=70, alpha=0.8, label=label,
                   edgecolors="black", linewidths=0.4)
    # chance reference: a random ranker's AP approx equals query prevalence
    lim = 0.75
    ax.plot([0, lim], [0, lim], color="grey", linestyle="dashed", linewidth=1.2,
            label="random-AP baseline (AP $\\approx$ prevalence)")
    # annotate the salient extremes
    for label, (marker, pts) in series.items():
        for prev, ap, q in pts:
            if ap > 0.55 or (ap - prev) > 0.30:
                ax.annotate(_short(q, 16), (prev, ap), fontsize=6.5,
                            xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("query prevalence (positive fraction)")
    ax.set_ylabel("zero-shot average precision (GeoRSCLIP)")
    ax.set_title("The salience law: recovered open-vocabulary signal scales with\n"
                 "change salience; points above the diagonal beat chance")
    ax.set_xlim(0, lim); ax.set_ylim(0, 0.9)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    return _save(fig, out_dir, "salience_law__summary", svg)


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-dataset retrieval figures from results/*.json")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="report/figures")
    ap.add_argument("--svg", action="store_true")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    od = Path(args.out_dir)
    made: List[Path] = []
    for ds in ("levir_cc", "second_cc"):
        p = fig_per_query_ap(rd, od, dataset=ds, svg=args.svg)
        if p:
            made.append(p)
    for fn in (fig_qfabric_status_map, fig_salience_law):
        p = fn(rd, od, svg=args.svg)
        if p:
            made.append(p)

    print(f"\nWrote {len(made)} figure(s) -> {od}/")
    for p in made:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
