"""
Controlled JPEG-vs-native ablation on the NATIVE 3 m Planet-Fusion DEN source.

Why this exists
---------------
The report's native-raster section (``sec:nativeraster``) compares the native 3 m
rasters against the preprocessed RGB/NIR-JPEG subset, but flags that the head-to-head
is **not fully controlled** — the two corpora differ in AOI count (23 vs 75) and
colour composite (native RGB vs JPEG NRG). This script removes those confounds: it
takes the **same** native rasters, the **same** AOIs/pairs, the **same** encoder,
colour composite, approach and CV folds, and varies **only** the JPEG compression
quality. Native (lossless) is the upper bound; each JPEG quality is the identical
corpus pushed through a lossy round-trip in memory before encoding.

So the only thing that changes between rows is the image degradation — this is the
clean ablation the report's caveat asks for.

What it does
------------
For ``--qualities q1 q2 ...`` (plus the native lossless baseline):
  1. wrap the registry-resolved ``dynamic_earthnet_planet`` loader so each tile is
     JPEG round-tripped at quality ``q`` (PIL -> JPEG bytes -> PIL) before encoding;
  2. encode every pair with the chosen encoder into an isolated, quality-tagged
     cache (``data/cache/...__abl_*``) — never touches the report's caches;
  3. run the **identical** zero-shot AOI k-fold CV used by ``cv_eval_planet3m.py``.

Relevance labels (from the lossless masks) and the text-query vectors are computed
**once** and reused across every quality, so they cannot drift between rows. Folds
depend only on the AOI set + seed, so the partition is identical too.

Run::

    uv run python feature_3m_native/jpeg_ablation.py \
        --root /media/markos/<drive>/dynamic_earthnet --folds 5 \
        --qualities 95 75 50 25 10
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Run from anywhere: put the repo root (this file's grandparent) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.benchmark import _average_precision, encode_query
from src.datasets.registry import build_dataset
from src.embeddings import PairEmbeddingStore, load_or_compute
from src.encoders import get_encoder
from src.queries import get_queries
from src.retrieval import ChangeRetriever
from src.stats import aoi_folds, rank_order

DATASET = "dynamic_earthnet_planet"


def _std(x) -> float:
    """Sample standard deviation (ddof=1); 0.0 for fewer than two values."""
    a = np.asarray(x, dtype=np.float64)
    return float(np.std(a, ddof=1)) if a.size > 1 else 0.0


class JpegDegraded:
    """Wraps a ``TemporalDataset`` and JPEG-round-trips every image it serves.

    Delegates the whole protocol to the inner loader (pair list, labels, name) and
    only intercepts pixel delivery, so the corpus, labels and pair ordering are
    byte-for-byte the inner loader's — the *only* change is lossy re-encoding.
    """

    def __init__(self, inner, quality: int, downsample: Optional[int] = None) -> None:
        self.inner = inner
        self.quality = quality
        self.downsample = downsample  # target px (square) before JPEG, or None
        self.name = inner.name  # keep the registry name so queries/labels resolve

    def list_pairs(self):
        return self.inner.list_pairs()

    def get_pair_label(self, pair):
        return self.inner.get_pair_label(pair)

    def _degrade(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        if self.downsample:
            # Bicubic downsample to mimic the preprocessed subset's resize step;
            # the encoder then upsamples to its own input size, so high-frequency
            # detail lost here is gone for good — this is the resolution test.
            img = img.resize((self.downsample, self.downsample), Image.BICUBIC)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def load_pair_images(self, pair) -> Tuple[Image.Image, Image.Image]:
        a, b = self.inner.load_pair_images(pair)
        return self._degrade(a), self._degrade(b)


def _sub_store(store: PairEmbeddingStore, idx) -> PairEmbeddingStore:
    return PairEmbeddingStore(
        dataset_name=store.dataset_name, encoder_name=store.encoder_name,
        embed_dim=store.embed_dim, pairs=[store.pairs[i] for i in idx],
        f_t1=store.f_t1[idx], f_t2=store.f_t2[idx])


def cv_macro(store, rel_all, tvec, evaluable, folds, seed, approach, enc):
    """Identical zero-shot AOI k-fold CV to ``cv_eval_planet3m.py``: partition AOIs
    into disjoint folds, score each fold independently, macro-average per fold."""
    pairs = store.pairs
    aois = sorted({p.location_id for p in pairs})
    aoi_fold = aoi_folds(aois, folds, seed)
    fold_of = np.array([aoi_fold[p.location_id] for p in pairs])
    fold_macro: List[float] = []
    per_q = {q.text: [] for q in evaluable}
    for k in range(folds):
        idx = np.where(fold_of == k)[0]
        sub = _sub_store(store, idx)
        rsub = ChangeRetriever(sub, enc)
        aps_this = []
        for q in evaluable:
            rel = rel_all[q.text][idx]
            if rel.sum() == 0:
                continue
            sc = rsub.score_vec(tvec[q.text], approach=approach)
            ap = _average_precision(rel[rank_order(sc, rel)])
            per_q[q.text].append(ap)
            aps_this.append(ap)
        fold_macro.append(float(np.mean(aps_this)) if aps_this else 0.0)
    return {
        "macro_mAP_mean": round(float(np.mean(fold_macro)), 4),
        "macro_mAP_std": round(_std(fold_macro), 4),
        "fold_macro": [round(x, 4) for x in fold_macro],
        "per_query": {qt: {"mean": round(float(np.mean(v)), 4), "std": round(_std(v), 4),
                           "n_folds": len(v)} for qt, v in per_q.items() if v},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Controlled JPEG-vs-native 3 m ablation")
    ap.add_argument("--root", default="data/dynamic_earthnet_planet",
                    help="dir holding labels.zip + planet.<UTM>.zip archives")
    ap.add_argument("--encoder", default="clip_vitl14")
    ap.add_argument("--color-mode", default="rgb", choices=["rgb", "nrg"])
    ap.add_argument("--qualities", type=int, nargs="+", default=[95, 75, 50, 25, 10],
                    help="JPEG quality levels to compare against native (lossless)")
    ap.add_argument("--downsample", type=int, nargs="*", default=[],
                    help="extra rows: bicubic-downsample to each px BEFORE JPEG, to "
                         "mimic the preprocessed subset's resize+compress pipeline")
    ap.add_argument("--downsample-quality", type=int, default=75,
                    help="JPEG quality applied to the downsampled rows (default 75)")
    ap.add_argument("--approach", default="zero_shot", choices=["zero_shot", "naive"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--results-dir", default="feature_3m_native/results")
    args = ap.parse_args()

    enc = get_encoder(args.encoder)
    color = args.color_mode
    csuf = "" if color == "rgb" else f"_{color}"

    # Native (lossless) corpus defines the shared pair set, labels and queries.
    ds_native = build_dataset(DATASET, root=args.root, split=None, color_mode=color)
    print(f"Encoding native (lossless) baseline ...")
    store_native = load_or_compute(ds_native, enc, cache_dir=args.cache_dir,
                                   cache_tag=f"all{csuf}__abl_native")

    queries = get_queries(DATASET)
    pairs = store_native.pairs
    rel_all = {q.text: np.array([bool((lb := ds_native.get_pair_label(p)) is not None
                                      and q.predicate(lb)) for p in pairs])
               for q in queries}
    evaluable = [q for q in queries if rel_all[q.text].sum() > 0]
    tvec = {q.text: encode_query(enc, q.text) for q in evaluable}
    print(f"corpus: {len(pairs)} pairs, {len({p.location_id for p in pairs})} AOIs, "
          f"{len(evaluable)} evaluable queries (labels/queries shared across all rows)")

    def _check(store) -> None:
        assert [tuple(p) for p in store.pairs] == [tuple(p) for p in pairs], \
            "pair set drifted between native and JPEG encode — comparison would not be controlled"

    rows = []
    native_cv = cv_macro(store_native, rel_all, tvec, evaluable, args.folds, args.seed,
                         args.approach, enc)
    rows.append({"source": "native", "quality": None, **native_cv})
    print(f"  native            macro mAP = {native_cv['macro_mAP_mean']} "
          f"± {native_cv['macro_mAP_std']}")

    for q in args.qualities:
        ds_jpeg = JpegDegraded(build_dataset(DATASET, root=args.root, split=None,
                                             color_mode=color), quality=q)
        print(f"Encoding JPEG quality={q} ...")
        store_q = load_or_compute(ds_jpeg, enc, cache_dir=args.cache_dir,
                                  cache_tag=f"all{csuf}__abl_jpeg{q}")
        _check(store_q)
        cv = cv_macro(store_q, rel_all, tvec, evaluable, args.folds, args.seed,
                      args.approach, enc)
        delta = round(cv["macro_mAP_mean"] - native_cv["macro_mAP_mean"], 4)
        rows.append({"source": f"jpeg_q{q}", "quality": q, "delta_vs_native": delta, **cv})
        print(f"  jpeg q={q:<3d}        macro mAP = {cv['macro_mAP_mean']} "
              f"± {cv['macro_mAP_std']}  (Δ vs native {delta:+.4f})")

    dq = args.downsample_quality
    for px in args.downsample:
        ds_d = JpegDegraded(build_dataset(DATASET, root=args.root, split=None,
                                          color_mode=color), quality=dq, downsample=px)
        print(f"Encoding downsample={px}px + JPEG q={dq} ...")
        store_d = load_or_compute(ds_d, enc, cache_dir=args.cache_dir,
                                  cache_tag=f"all{csuf}__abl_down{px}q{dq}")
        _check(store_d)
        cv = cv_macro(store_d, rel_all, tvec, evaluable, args.folds, args.seed,
                      args.approach, enc)
        delta = round(cv["macro_mAP_mean"] - native_cv["macro_mAP_mean"], 4)
        rows.append({"source": f"down{px}", "quality": None, "downsample_px": px,
                     "jpeg_quality": dq, "delta_vs_native": delta, **cv})
        print(f"  down {px}px q{dq}      macro mAP = {cv['macro_mAP_mean']} "
              f"± {cv['macro_mAP_std']}  (Δ vs native {delta:+.4f})")

    out = {
        "dataset": DATASET, "encoder": args.encoder, "color_mode": color,
        "approach": args.approach, "folds": args.folds, "seed": args.seed,
        "n_pairs": len(pairs), "n_aois": len({p.location_id for p in pairs}),
        "n_evaluable_queries": len(evaluable),
        "control": ("identical AOIs/pairs/colour/encoder/approach/folds; only JPEG "
                    "quality varies. Labels (lossless masks) and query vectors shared "
                    "across all rows."),
        "rows": rows,
    }
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    op = Path(args.results_dir) / f"jpeg_ablation__{args.encoder}__{color}__{args.approach}.json"
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
