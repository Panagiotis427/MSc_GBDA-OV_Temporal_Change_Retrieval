"""
Temporal pinpointing on the native 3 m Planet-Fusion DEN source.

The project brief singles out Dynamic EarthNet as "ideal for testing the model's
ability to pinpoint exact time-steps of change", and asks evaluation to "verify if
the model correctly identifies the specific temporal window where the change was
most prominent compared to stable time-steps". The repository's retrieval metrics
answer "which *pair* shows the change" (ranking across AOIs); this script answers
the orthogonal, brief-mandated question "*when* does the change happen" (ranking
time-steps *within* one AOI's timeline).

Construction (reuses the existing scoring, only the evaluation axis is new):
  * build the loader with monthly pairing, so each AOI's consecutive pairs are its
    timeline steps (month_{t-1} -> month_t);
  * the per-step change score is the standard zero-shot Δ-similarity for that pair,
    cos(q, f_t) - cos(q, f_{t-1}) (``ChangeRetriever.score_vec(..., "zero_shot")``);
  * a step is *relevant* if the query predicate fires on its PairLabel (the true
    transition month for that change type);
  * within each AOI we rank the steps by change score and score the ranking with
    average precision -> a per-AOI temporal AP. We also report the peak-hit rate
    (is the top-scored step a true transition?) at exact and ±1-month tolerance.

Honesty machinery matches the rest of the repo: per-query permutation p-value
(shuffling relevance *within* each AOI, preserving its step count) and BH-FDR.

Run::

    uv run python feature_3m_native/temporal_pinpoint.py \
        --root /media/markos/<drive>/dynamic_earthnet --encoder clip_vitl14
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.benchmark import _average_precision, encode_query
from src.datasets.registry import build_dataset
from src.embeddings import cache_tag_for, load_or_compute
from src.encoders import get_encoder
from src.queries import get_queries
from src.retrieval import ChangeRetriever
from src.stats import bh_fdr, perm_p_value, rank_order

DATASET = "dynamic_earthnet_planet"
PERM = 2000


def _std(x) -> float:
    a = np.asarray(x, dtype=np.float64)
    return float(np.std(a, ddof=1)) if a.size > 1 else 0.0


def _aoi_order(pairs):
    """Map AOI -> step indices into the pair store, in chronological order."""
    byaoi = defaultdict(list)
    for i, p in enumerate(pairs):
        byaoi[p.location_id].append(i)
    for loc in byaoi:
        byaoi[loc].sort(key=lambda i: pairs[i].t1_key)
    return byaoi


def main() -> None:
    ap = argparse.ArgumentParser(description="Temporal pinpointing on native 3 m DEN")
    ap.add_argument("--root", default="data/dynamic_earthnet_planet")
    ap.add_argument("--encoder", default="clip_vitl14")
    ap.add_argument("--color-mode", default="rgb", choices=["rgb", "nrg"])
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--results-dir", default="feature_3m_native/results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    enc = get_encoder(args.encoder)
    color = args.color_mode

    ds = build_dataset(DATASET, root=args.root, split=None, color_mode=color,
                       pairing="monthly")
    store = load_or_compute(ds, enc, cache_dir=args.cache_dir,
                            cache_tag=cache_tag_for("all", color) + "__monthly")
    pairs = store.pairs
    assert [tuple(p) for p in pairs] == [tuple(p) for p in ds.list_pairs()]
    byaoi = _aoi_order(pairs)
    retr = ChangeRetriever(store, enc)

    queries = get_queries(DATASET)
    # Per-step relevance per query (predicate on each consecutive-month PairLabel).
    rel_all = {}
    for q in queries:
        rel = np.zeros(len(pairs), bool)
        for i, p in enumerate(pairs):
            lb = ds.get_pair_label(p)
            rel[i] = bool(lb is not None and q.predicate(lb))
        rel_all[q.text] = rel

    n_steps_total = len(pairs)
    print(f"timeline: {len(byaoi)} AOIs, {n_steps_total} monthly steps "
          f"(mean {n_steps_total / len(byaoi):.1f} steps/AOI)")

    results = []
    example = None  # best clean (query, AOI) timeline for the figure
    for q in queries:
        rel = rel_all[q.text]
        scores = retr.score_vec(encode_query(enc, q.text), approach="zero_shot")
        per_aoi = []          # (sc_aoi, rel_aoi) for AOIs with >=1 relevant step
        aps, peaks, peaks_pm1, prevs = [], [], [], []
        for loc, idx in byaoi.items():
            idx = np.array(idx)
            r = rel[idx]
            if len(idx) < 2 or r.sum() == 0:
                continue
            sc = scores[idx]
            ranked = rank_order(sc, r)
            apv = _average_precision(r[ranked])
            top = int(np.argmax(sc))
            hit = bool(r[top])
            hit_pm1 = bool(r[max(0, top - 1):top + 2].any())
            aps.append(apv)
            peaks.append(hit)
            peaks_pm1.append(hit_pm1)
            prevs.append(r.mean())
            per_aoi.append((sc, r))
            if hit and r.sum() == 1 and (example is None or apv > example["ap"]):
                example = {"query": q.text, "aoi": loc, "ap": apv,
                           "months": [pairs[i].t2_key for i in idx],
                           "scores": [round(float(x), 4) for x in sc],
                           "relevant_step": int(np.argmax(r))}
        if not aps:
            continue
        obs = float(np.mean(aps))
        # permutation: shuffle relevance within each AOI, preserve step counts
        null = np.empty(PERM)
        for b in range(PERM):
            vals = []
            for sc, r in per_aoi:
                rp = rng.permutation(r)
                vals.append(_average_precision(rp[rank_order(sc, rp)]))
            null[b] = np.mean(vals)
        p = perm_p_value(int(np.sum(null >= obs)), PERM)
        results.append({
            "query": q.text, "category": q.category,
            "n_aois": len(aps),
            "temporal_mAP": round(obs, 4),
            "rand_mAP": round(float(null.mean()), 4),
            "perm_p": round(p, 4),
            "peak_hit_rate": round(float(np.mean(peaks)), 4),
            "peak_hit_rate_pm1": round(float(np.mean(peaks_pm1)), 4),
            "mean_prevalence": round(float(np.mean(prevs)), 4),
        })

    for r, qv in zip(results, bh_fdr([r["perm_p"] for r in results])):
        r["bh_fdr"] = round(float(qv), 4)

    macro = round(float(np.mean([r["temporal_mAP"] for r in results])), 4)
    macro_rand = round(float(np.mean([r["rand_mAP"] for r in results])), 4)
    macro_peak = round(float(np.mean([r["peak_hit_rate"] for r in results])), 4)
    macro_peak_pm1 = round(float(np.mean([r["peak_hit_rate_pm1"] for r in results])), 4)
    n_sig = sum(1 for r in results if r["bh_fdr"] < 0.05)

    out = {
        "dataset": DATASET, "encoder": args.encoder, "color_mode": color,
        "axis": "temporal pinpointing (rank time-steps within each AOI timeline)",
        "score": "zero-shot Δ-similarity per monthly step",
        "n_aois": len(byaoi), "n_steps": n_steps_total,
        "macro_temporal_mAP": macro, "macro_rand_mAP": macro_rand,
        "macro_peak_hit_rate": macro_peak, "macro_peak_hit_rate_pm1": macro_peak_pm1,
        "n_queries_evaluable": len(results), "n_fdr_significant": n_sig,
        "per_query": results, "example_timeline": example,
    }
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    op = Path(args.results_dir) / f"temporal_pinpoint__{args.encoder}__{color}.json"
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nmacro temporal mAP = {macro} (random {macro_rand}); "
          f"peak-hit {macro_peak} (±1mo {macro_peak_pm1}); "
          f"{n_sig}/{len(results)} queries FDR-significant")
    for r in results:
        print(f"  tAP={r['temporal_mAP']:.3f} (rand {r['rand_mAP']:.3f}) "
              f"peak={r['peak_hit_rate']:.2f}/±1 {r['peak_hit_rate_pm1']:.2f} "
              f"p={r['perm_p']:.3f} q={r['bh_fdr']:.3f} n={r['n_aois']:2d}  {r['query'][:42]}")
    print(f"wrote {op}")


if __name__ == "__main__":
    main()
