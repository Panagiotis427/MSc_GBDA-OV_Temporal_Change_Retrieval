"""
Static zero-shot vs PEFT visual comparison — the qualitative half of the graded
comparison.

For a few representative DEN queries it renders, side by side, the top-K pairs
retrieved by ``zero_shot`` and by ``peft``. Each cell shows the pair's
[T1 | T2] thumbnails, the rank + score, and a green/red border for
relevant/irrelevant (from the query's label predicate). One PNG per encoder ->
``figures/zeroshot_vs_peft__<encoder>__<split>.png``.

Embeddings are read from cache (no image forward passes); only the cheap *text*
tower runs. Defaults to ``--split train`` where PEFT visibly wins (it overfits
on held-out splits — see REPORT 7.2).

Run::

    python -m scripts.make_comparison_figure --encoder clip_vitl14 --split train --top-k 3
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.datasets.registry import build_dataset  # noqa: E402
from src.embeddings import cache_tag_for, color_tag, load_or_compute  # noqa: E402
from src.encoders import get_encoder  # noqa: E402
from src.model import load_adapter  # noqa: E402
from src.queries import get_queries  # noqa: E402
from src.retrieval import ChangeRetriever  # noqa: E402

# Substrings matched against src/queries/den.py texts -> representative queries.
_REPRESENTATIVE = ["buildings", "deforestation", "snow"]


def representative_queries(all_queries, wanted=None) -> List:
    wanted = wanted or _REPRESENTATIVE
    picked = []
    for w in wanted:
        for q in all_queries:
            if w in q.text.lower() and q not in picked:
                picked.append(q)
                break
    return picked or list(all_queries[:3])


def _pair_thumb(ds, pair, size: int = 96) -> np.ndarray:
    """[T1 | T2] horizontally-concatenated RGB uint8 array."""
    t1, t2 = ds.load_pair_images(pair)
    arrs = []
    for im in (t1, t2):
        a = np.asarray(im.convert("RGB").resize((size, size)))
        arrs.append(a)
    gap = np.full((size, 4, 3), 255, dtype=np.uint8)
    return np.concatenate([arrs[0], gap, arrs[1]], axis=1)


def render(ds, retr: ChangeRetriever, adapter, out_dir, *, encoder: str,
           split: str, queries=None, top_k: int = 3, svg: bool = False) -> Optional[Path]:
    if queries is None:
        queries = representative_queries(get_queries(ds.name))
    if not queries:
        print("  skip comparison: no queries")
        return None
    approaches = ["zero_shot"] + (["peft"] if adapter is not None else [])
    nrows = len(queries) * len(approaches)
    fig, axes = plt.subplots(nrows, top_k,
                             figsize=(2.4 * top_k, 1.8 * nrows + 0.5),
                             squeeze=False)
    row = 0
    for q in queries:
        for appr in approaches:
            if appr == "peft":
                retr.set_adapter(adapter, feature_mode="difference")
            results = retr.search(q.text, approach=appr, top_k=top_k)
            for col in range(top_k):
                ax = axes[row][col]
                ax.set_xticks([]); ax.set_yticks([])
                if col < len(results):
                    res = results[col]
                    ax.imshow(_pair_thumb(ds, res.pair))
                    lb = ds.get_pair_label(res.pair)
                    relevant = bool(lb is not None and q.predicate(lb))
                    color = "#2ca02c" if relevant else "#d62728"
                    for sp in ax.spines.values():
                        sp.set_edgecolor(color); sp.set_linewidth(3)
                    ax.set_title(f"#{res.rank+1}  s={res.score:.2f}", fontsize=8)
                if col == 0:
                    ax.set_ylabel(f"{q.text[:22]}\n[{appr}]", fontsize=7,
                                  rotation=0, ha="right", va="center", labelpad=28)
            row += 1
    fig.suptitle(f"Zero-shot vs PEFT — top-{top_k}  ({encoder}, split={split})\n"
                 "[T1 | T2];  green=relevant, red=not", fontsize=10)
    fig.tight_layout(rect=[0.04, 0, 1, 0.96])
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    name = f"zeroshot_vs_peft__{encoder}__{split}"
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if svg:
        fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    return path


def load_engine(encoder: str, split: str, color: str, *, root: str, cache_dir: str):
    """Build (dataset, retriever, adapter) from cached embeddings + adapter."""
    enc = get_encoder(encoder)
    ds = build_dataset("dynamic_earthnet", root=root, pairing="bimonthly",
                       split=split, color_mode=color)
    store = load_or_compute(ds, enc, cache_dir=cache_dir,
                            cache_tag=cache_tag_for(split, color))
    retr = ChangeRetriever(store, enc)
    apath = Path("models") / f"dynamic_earthnet__{encoder}{color_tag(color)}__adapter.pt"
    adapter = None
    if apath.exists():
        adapter, _ = load_adapter(str(apath))
    else:
        print(f"  note: no adapter at {apath} -> zero_shot only")
    return ds, retr, adapter


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-shot vs PEFT comparison figure")
    ap.add_argument("--encoder", default="clip_vitl14",
                    choices=["clip_vitl14", "georsclip", "remoteclip"])
    ap.add_argument("--root", default="data/DynamicEarthNet")
    ap.add_argument("--split", default="train")
    ap.add_argument("--color", default="rgb", choices=["rgb", "nrg", "ndvi"])
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out-dir", default="latex/figures")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--svg", action="store_true")
    args = ap.parse_args()

    ds, retr, adapter = load_engine(args.encoder, args.split, args.color,
                                    root=args.root, cache_dir=args.cache_dir)
    path = render(ds, retr, adapter, args.out_dir, encoder=args.encoder,
                  split=args.split, top_k=args.top_k, svg=args.svg)
    if path:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
