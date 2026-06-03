"""
S3 — patch-level (localised) change retrieval, vs the global-embedding baseline.

REPORT Appendix B.9 showed the residual DEN weakness is a *method* ceiling: global
CLIP-embedding differencing averages over the whole 1024² tile, so localised change
(a new building, a cleared patch) barely moves the embedding. This script tests the
fix: score each pair from **per-patch** embeddings instead of one global vector.

For a query ``t`` and a pair with per-patch embeddings ``P1, P2`` (``[n_patch, D]``,
spatially aligned T1/T2 grids):

* ``patch_naive``   : ``max_p cos(t, P2_p)``               — localised end-state match.
* ``patch_zeroshot``: ``max_p ( cos(t,P2_p) - cos(t,P1_p) )`` — the patch whose
                      similarity to the query *grew most* T1→T2 (localised change).
* ``patch_top3``    : mean of the top-3 per-patch deltas (less point-sensitive).

Evaluated on the full 75-AOI corpus (bootstrap CI + permutation p) and under 5-fold
AOI cross-validation, with **fraction-based relevance** (the curated labels from
B.9), so results are directly comparable to ``cv_eval.py --relevance fraction``.

Patch embeddings are encoded once and cached (encoding 1650 images is the only GPU
cost; re-runs over scoring variants are instant).

Run::

    python -m scripts.patch_eval --encoder georsclip --color-mode nrg --folds 5
    python -m scripts.patch_eval --encoder clip_vitl14 --color-mode nrg --folds 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.benchmark import _average_precision, encode_query
from src.datasets.dynamic_earthnet_pp import DENNpyDataset
from src.encoders import get_encoder
from src.queries.den import frac_queries
from scripts.cv_eval import _merge_stores

BOOTSTRAP = 1000
PERM = 4000


def _encode_patches(ds, enc, cache_dir, enc_name, color):
    """Encode (or load) per-pair patch embeddings → P1, P2 arrays [N, n_patch, D]."""
    cache = Path(cache_dir) / f"patch__{enc_name}__{color}.npz"
    pairs = ds.list_pairs()
    if cache.exists():
        d = np.load(cache)
        if int(d["n"]) == len(pairs):
            print(f"loaded patch cache {cache} (N={int(d['n'])})")
            return d["p1"], d["p2"], pairs
    print(f"encoding {len(pairs)} pairs to patch embeddings ({enc_name}, {color})...")
    p1, p2 = [], []
    for i, pk in enumerate(pairs):
        im1, im2 = ds.load_pair_images(pk)
        a = enc.encode_image_patches(im1)
        b = enc.encode_image_patches(im2)
        if a is None or b is None:
            raise RuntimeError(f"{enc_name} does not expose patch tokens")
        p1.append(a)
        p2.append(b)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(pairs)}")
    p1 = np.stack(p1).astype(np.float32)
    p2 = np.stack(p2).astype(np.float32)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, p1=p1, p2=p2, n=len(pairs))
    print(f"cached patch embeddings -> {cache}  shape {p1.shape}")
    return p1, p2, pairs


def _z(x):
    s = x.std()
    return (x - x.mean()) / (s + 1e-8)


def _smooth3x3(grid):
    """3x3 mean smoothing of a [N, gh, gw] map with edge replication
    (no SciPy). Rewards spatially-contiguous change over single-patch spikes."""
    p = np.pad(grid, ((0, 0), (1, 1), (1, 1)), mode="edge")
    acc = np.zeros_like(grid, dtype=np.float64)
    for di in range(3):
        for dj in range(3):
            acc += p[:, di:di + grid.shape[1], dj:dj + grid.shape[2]]
    return (acc / 9.0).astype(np.float32)


def _patch_score(P1, P2, t, approach, tau: float = 0.03):
    s2 = P2 @ t            # [N, n_patch]
    if approach == "patch_naive":
        return s2.max(axis=1)
    s1 = P1 @ t
    delta = s2 - s1        # [N, n_patch]
    if approach == "patch_top3" or approach == "hybrid":
        k = min(3, delta.shape[1])
        return np.sort(delta, axis=1)[:, -k:].mean(axis=1)
    if approach == "patch_zeroshot":
        return delta.max(axis=1)
    if approach == "patch_softattn":
        # query-conditioned soft-attention: softmax(delta/tau)-weighted mean of delta.
        # tau->0 recovers max, tau->inf recovers mean; a middle tau is a soft top-k.
        w = np.exp((delta - delta.max(axis=1, keepdims=True)) / tau)
        w /= w.sum(axis=1, keepdims=True)
        return (w * delta).sum(axis=1)
    if approach == "patch_spatial":
        n = delta.shape[1]
        side = int(round(n ** 0.5))
        if side * side != n:                       # non-square grid -> fall back
            k = min(3, n)
            return np.sort(delta, axis=1)[:, -k:].mean(axis=1)
        sm = _smooth3x3(delta.reshape(-1, side, side)).reshape(delta.shape[0], n)
        k = min(3, n)
        return np.sort(sm, axis=1)[:, -k:].mean(axis=1)
    raise ValueError(f"unknown approach {approach!r}")


def _scores(P1, P2, t, approach, G1=None, G2=None, tau: float = 0.03):
    """Per-pair score. For ``hybrid``, fuse the global Δ-cosine and the
    patch top-3 Δ by z-scoring each over the candidate set and summing
    (rank-comparable; diffuse change favours global, localised favours patch)."""
    if approach == "hybrid":
        patch = _patch_score(P1, P2, t, "patch_top3")
        glob = (G2 @ t) - (G1 @ t)
        return _z(glob) + _z(patch)
    return _patch_score(P1, P2, t, approach, tau=tau)


def _rand_ap(R, N, rng):
    rel = np.zeros(N, bool)
    rel[:R] = True
    rng.shuffle(rel)
    hits = np.cumsum(rel)
    return float((hits / np.arange(1, N + 1))[rel].sum() / R)


def _perm_p(N, n_rel, obs, rng, iters=PERM):
    base = np.zeros(N, bool)
    base[:n_rel] = True
    draws = np.array([_rand_ap(n_rel, N, rng) for _ in range(iters)])
    return float((draws >= obs).mean()), float(draws.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch-level (localised) change retrieval eval")
    ap.add_argument("--encoder", default="georsclip")
    ap.add_argument("--color-mode", default="nrg", choices=["rgb", "nrg", "ndvi"])
    ap.add_argument("--root", default="data/DynamicEarthNet")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--approach", default="patch_zeroshot",
                    choices=["patch_zeroshot", "patch_naive", "patch_top3", "hybrid",
                             "patch_softattn", "patch_spatial"])
    ap.add_argument("--tau", type=float, default=0.03,
                    help="softmax temperature for --approach patch_softattn")
    ap.add_argument("--prompt-ensemble", action="store_true",
                    help="ensemble the query text embedding over prompt templates")
    ap.add_argument("--frac-thresh", type=float, default=0.05)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    enc = get_encoder(args.encoder)
    ds = DENNpyDataset(root=args.root, split=None, color_mode=args.color_mode)
    P1, P2, pairs = _encode_patches(ds, enc, args.cache_dir, args.encoder, args.color_mode)
    N = len(pairs)
    print(f"corpus {N} pairs, patch grid {P1.shape[1]}, D={P1.shape[2]}"
          + (" | prompt-ensemble ON" if args.prompt_ensemble else ""))

    # Global embeddings (for the hybrid approach), aligned to `pairs` order.
    G1 = G2 = None
    if args.approach == "hybrid":
        gstore = _merge_stores("dynamic_earthnet", args.encoder, args.color_mode, args.cache_dir)
        row = {(p.location_id, p.t1_key, p.t2_key): i for i, p in enumerate(gstore.pairs)}
        idx = np.array([row[(p.location_id, p.t1_key, p.t2_key)] for p in pairs])
        norm = lambda a: a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
        G1, G2 = norm(gstore.f_t1[idx]), norm(gstore.f_t2[idx])

    queries = frac_queries(args.frac_thresh)
    labels = [ds.get_pair_label(p) for p in pairs]
    rel_all = {q.text: np.array([bool(lb is not None and q.predicate(lb)) for lb in labels])
               for q in queries}
    evaluable = [q for q in queries if rel_all[q.text].sum() > 0]
    tvecs = {q.text: encode_query(enc, q.text, ensemble=args.prompt_ensemble) for q in evaluable}

    # full corpus
    full = []
    for q in evaluable:
        rel = rel_all[q.text]
        sc = _scores(P1, P2, tvecs[q.text], args.approach, G1, G2, tau=args.tau)
        ap_obs = _average_precision(rel[np.argsort(-sc)])
        boot = np.empty(BOOTSTRAP)
        for b in range(BOOTSTRAP):
            samp = rng.integers(0, N, N)
            r = rel[samp]
            boot[b] = 0.0 if r.sum() == 0 else _average_precision(r[np.argsort(-sc[samp])])
        p, randmean = _perm_p(N, int(rel.sum()), ap_obs, rng)
        full.append({"query": q.text, "n_relevant": int(rel.sum()), "ap": round(ap_obs, 4),
                     "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                              round(float(np.percentile(boot, 97.5)), 4)],
                     "rand_ap": round(randmean, 4), "perm_p": round(p, 4)})
    macro_full = round(float(np.mean([r["ap"] for r in full])), 4) if full else 0.0

    # k-fold AOI CV
    aois = sorted({p.location_id for p in pairs})
    perm_aois = list(rng.permutation(aois))
    aoi_fold = {a: i % args.folds for i, a in enumerate(perm_aois)}
    fold_of = np.array([aoi_fold[p.location_id] for p in pairs])
    fold_macro = []
    for k in range(args.folds):
        idx = np.where(fold_of == k)[0]
        aps = []
        for q in evaluable:
            rel = rel_all[q.text][idx]
            if rel.sum() == 0:
                continue
            sc = _scores(P1[idx], P2[idx], tvecs[q.text], args.approach,
                         None if G1 is None else G1[idx], None if G2 is None else G2[idx],
                         tau=args.tau)
            aps.append(_average_precision(rel[np.argsort(-sc)]))
        fold_macro.append(float(np.mean(aps)) if aps else 0.0)

    out = {"dataset": "dynamic_earthnet", "encoder": args.encoder, "color_mode": args.color_mode,
           "approach": args.approach, "relevance": "fraction", "frac_thresh": args.frac_thresh,
           "prompt_ensemble": args.prompt_ensemble, "tau": args.tau,
           "n_pairs": N, "patch_grid": int(P1.shape[1]), "n_evaluable_queries": len(evaluable),
           "full_corpus": {"macro_mAP": macro_full, "per_query": full},
           "kfold": {"macro_mAP_mean": round(float(np.mean(fold_macro)), 4),
                     "macro_mAP_std": round(float(np.std(fold_macro)), 4),
                     "fold_macro": [round(x, 4) for x in fold_macro]}}
    ens = "__ens" if args.prompt_ensemble else ""
    op = (Path(args.results_dir) /
          f"patch_eval__{args.encoder}__{args.color_mode}__{args.approach}{ens}.json")
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[{args.approach}] full-corpus macro mAP = {macro_full}  "
          f"| k-fold = {out['kfold']['macro_mAP_mean']} ± {out['kfold']['macro_mAP_std']}")
    for r in full:
        sig = "*" if r["perm_p"] < 0.05 else " "
        print(f" {sig} ap={r['ap']:.3f} CI{r['ci95']} rand={r['rand_ap']:.3f} "
              f"p={r['perm_p']:.3f} n={r['n_relevant']:3d} {r['query'][:42]}")
    print(f"wrote {op}")


if __name__ == "__main__":
    main()
