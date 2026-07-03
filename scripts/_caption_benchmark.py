"""
Shared driver for the caption-derived open-vocabulary retrieval benchmarks
(LEVIR-CC, SECOND-CC).

Both datasets run the identical flow — embed (or load cached) per-pair embeddings,
score the caption-grounded query set (``src/queries/<name>.py``) under the naive
and zero-shot approaches for each frozen encoder, and write one JSON per encoder
to ``results/`` with per-query AP. The only differences are the dataset name, the
default root, and the display label, so the body lives here once (the two entry
scripts previously diverged — e.g. on the cache-tag construction).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from src.benchmark import run_benchmark
from src.datasets.registry import build_dataset
from src.embeddings import cache_tag_for, load_or_compute
from src.encoders import get_encoder
from src.queries import get_queries
from src.retrieval import ChangeRetriever


def run_caption_benchmark(
    dataset_name: str,
    *,
    display_name: str,
    default_root: str,
    root_help: str,
    argv: Optional[Sequence[str]] = None,
) -> None:
    ap = argparse.ArgumentParser(
        description=f"{display_name} open-vocab retrieval benchmark")
    ap.add_argument("--root", default=default_root, help=root_help)
    ap.add_argument("--split", default="test")
    ap.add_argument("--encoders", nargs="+",
                    default=["georsclip", "clip_vitl14", "remoteclip"])
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args(argv)

    ds = build_dataset(dataset_name, root=args.root, split=args.split)
    pairs = ds.list_pairs()
    labels = [ds.get_pair_label(p) for p in pairs]
    n = len(pairs)
    queries = get_queries(dataset_name)
    prevalence = {q.text: round(sum(1 for lb in labels if q.predicate(lb)) / n, 4)
                  for q in queries}
    macro_prev = round(sum(prevalence.values()) / len(prevalence), 4)
    print(f"{display_name} {args.split}: {n} pairs | macro prevalence baseline = {macro_prev}")
    for q in queries:
        print(f"  prevalence {prevalence[q.text]:.3f}  {q.text}")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    for enc_name in args.encoders:
        enc = get_encoder(enc_name)
        store = load_or_compute(ds, enc, cache_dir=args.cache_dir,
                                cache_tag=cache_tag_for(args.split, "rgb"))
        retr = ChangeRetriever(store, enc)
        out = {"dataset": dataset_name, "encoder": enc_name, "split": args.split,
               "color_mode": "rgb", "n_pairs": n,
               "macro_prevalence_baseline": macro_prev,
               "query_prevalence": prevalence, "approaches": {}}
        for approach in ("naive", "zero_shot"):
            rep = run_benchmark(ds, retr, approach=approach)
            out["approaches"][approach] = {
                "macro_mAP": round(float(rep.mAP), 4),
                "per_query_ap": {q.text: round(float(q.ap), 4) for q in rep.per_query},
                "per_query_n_relevant": {q.text: int(q.n_relevant) for q in rep.per_query},
            }
            print(f"{enc_name:11} {approach:9} macro_mAP={rep.mAP:.4f}")
            for q in rep.per_query:
                print(f"    AP={q.ap:5.3f}  nrel={q.n_relevant:4d}  {q.text}")
        op = Path(args.results_dir) / f"{dataset_name}__{enc_name}__{args.split}.json"
        op.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  wrote {op}")
