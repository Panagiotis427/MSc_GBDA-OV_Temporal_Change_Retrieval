"""
Cross-validated DEN retrieval evaluation with confidence intervals.

Motivation (REPORT Appendix B): the committed DEN **test** split (110 pairs)
makes only **3** wetland queries evaluable, so the headline 0.426 rests on n=3
and no generalisation across change-types can be claimed. This script removes
that single-split bottleneck two ways, **without re-encoding any imagery** — it
merges the cached train+val+test pair embeddings (= all 75 AOIs, 825 pairs):

1. **Full-corpus estimate.** Score every query over all 825 pairs → per-query AP
   with a **bootstrap 95% CI** (resampling pairs) and a **permutation p-value**
   vs random ranking. Pooling all AOIs makes 6 queries evaluable (vs 3 on test):
   the 3 wetland transitions + water-body + bare-soil + forest-loss.
2. **K-fold AOI cross-validation.** Partition the 75 AOIs into K disjoint folds.
   - zero_shot: evaluate each fold independently → per-query AP across folds →
     mean ± CI (variance over AOI samples, not over a single 3-query split).
   - peft (``--peft``): for each fold, train the adapter on the *other* folds'
     pairs and evaluate on the held-out fold — a leakage-free cross-validated
     PEFT estimate (contrast with the train-fit 0.42/0.998 of Appendix B.5).

Note: the 4 absent change-types (building / urban / deforestation / snow) have
**zero positives anywhere in this 75-AOI subset** — CV cannot conjure them; that
needs more diverse data (REPORT §10 / Appendix B, Tier B).

Run::

    python -m scripts.cv_eval --encoder georsclip --color-mode nrg --folds 5
    python -m scripts.cv_eval --encoder georsclip --color-mode rgb --folds 5 --peft
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.benchmark import _average_precision
from src.datasets.dynamic_earthnet_pp import DENNpyDataset
from src.embeddings import PairEmbeddingStore, cache_path
from src.encoders import get_encoder
from src.queries import get_queries
from src.retrieval import ChangeRetriever

SPLITS = ("train", "val", "test")
BOOTSTRAP = 1000
PERM = 4000


def _color_tag(split: str, color: str) -> str:
    return split if color == "rgb" else f"{split}_{color}"


def _merge_stores(dataset_name, encoder_name, color, cache_dir):
    """Concatenate the cached train/val/test stores into one 75-AOI corpus store."""
    pairs, f1, f2, dim = [], [], [], None
    for sp in SPLITS:
        p = cache_path(cache_dir, dataset_name, encoder_name, tag=_color_tag(sp, color))
        if not p.exists():
            raise FileNotFoundError(f"missing cache {p} — run the pipeline for split={sp} first")
        st = PairEmbeddingStore.load(p)
        pairs.extend(st.pairs)
        f1.append(st.f_t1)
        f2.append(st.f_t2)
        dim = st.embed_dim
    return PairEmbeddingStore(dataset_name=dataset_name, encoder_name=encoder_name,
                              embed_dim=dim, pairs=pairs,
                              f_t1=np.concatenate(f1), f_t2=np.concatenate(f2))


def _sub_store(store, idx):
    return PairEmbeddingStore(dataset_name=store.dataset_name, encoder_name=store.encoder_name,
                              embed_dim=store.embed_dim, pairs=[store.pairs[i] for i in idx],
                              f_t1=store.f_t1[idx], f_t2=store.f_t2[idx])


def _rel_vector(dataset, pairs, predicate):
    return np.array([bool((lb := dataset.get_pair_label(p)) is not None and predicate(lb))
                     for p in pairs])


def _perm_p(rel_ranked_len, n_rel, obs_ap, rng, iters=PERM):
    """P(random-ranking AP >= obs_ap) for a query with n_rel positives in N items."""
    N = rel_ranked_len
    draws = np.empty(iters)
    base = np.zeros(N, bool)
    base[:n_rel] = True
    for i in range(iters):
        perm = rng.permutation(N)
        rel = base[perm]
        hits = np.cumsum(rel)
        draws[i] = (hits / np.arange(1, N + 1))[rel].sum() / n_rel
    return float((draws >= obs_ap).mean()), float(draws.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-validated DEN retrieval eval with CIs")
    ap.add_argument("--encoder", default="georsclip")
    ap.add_argument("--color-mode", default="nrg", choices=["rgb", "nrg", "ndvi"])
    ap.add_argument("--root", default="data/DynamicEarthNet")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--approach", default="zero_shot", choices=["zero_shot", "naive"])
    ap.add_argument("--peft", action="store_true", help="also run leakage-free k-fold PEFT")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    enc = get_encoder(args.encoder)
    ds = DENNpyDataset(root=args.root, split=None, color_mode=args.color_mode)
    store = _merge_stores("dynamic_earthnet", args.encoder, args.color_mode, args.cache_dir)
    assert len(store) == len(ds.list_pairs()), (len(store), len(ds.list_pairs()))
    pairs = store.pairs
    queries = get_queries("dynamic_earthnet")
    aois = sorted({p.location_id for p in pairs})
    print(f"corpus: {len(pairs)} pairs, {len(aois)} AOIs, {len(queries)} queries registered")

    retr = ChangeRetriever(store, enc)
    rel_all = {q.text: _rel_vector(ds, pairs, q.predicate) for q in queries}
    evaluable = [q for q in queries if rel_all[q.text].sum() > 0]
    print(f"evaluable on full corpus: {len(evaluable)} queries")

    # --- 1. full-corpus per-query AP + bootstrap CI + permutation p ----------
    full = []
    N = len(pairs)
    for q in evaluable:
        rel = rel_all[q.text]
        scores = retr.score_all(q.text, approach=args.approach)
        order = np.argsort(-scores)
        ap_obs = _average_precision(rel[order])
        # bootstrap over pairs
        boot = np.empty(BOOTSTRAP)
        for b in range(BOOTSTRAP):
            samp = rng.integers(0, N, N)
            r = rel[samp]
            if r.sum() == 0:
                boot[b] = 0.0
                continue
            o = np.argsort(-scores[samp])
            boot[b] = _average_precision(r[o])
        p, randmean = _perm_p(N, int(rel.sum()), ap_obs, rng)
        full.append({"query": q.text, "n_relevant": int(rel.sum()), "ap": round(ap_obs, 4),
                     "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                              round(float(np.percentile(boot, 97.5)), 4)],
                     "rand_ap": round(randmean, 4), "perm_p": round(p, 4)})
    macro_full = round(float(np.mean([r["ap"] for r in full])), 4) if full else 0.0

    # --- 2. K-fold AOI CV ----------------------------------------------------
    perm_aois = list(rng.permutation(aois))
    folds = [perm_aois[i::args.folds] for i in range(args.folds)]
    aoi_fold = {a: k for k, fl in enumerate(folds) for a in fl}
    fold_of_pair = np.array([aoi_fold[p.location_id] for p in pairs])

    def fold_idx(k):
        return np.where(fold_of_pair == k)[0]

    # zero_shot CV: per-query AP per fold
    cv_zs = {q.text: [] for q in evaluable}
    fold_macro = []
    for k in range(args.folds):
        idx = fold_idx(k)
        sub = _sub_store(store, idx)
        rsub = ChangeRetriever(sub, enc)
        aps_this = []
        for q in evaluable:
            rel = rel_all[q.text][idx]
            if rel.sum() == 0:
                continue
            sc = rsub.score_all(q.text, approach=args.approach)
            ap_k = _average_precision(rel[np.argsort(-sc)])
            cv_zs[q.text].append(ap_k)
            aps_this.append(ap_k)
        fold_macro.append(float(np.mean(aps_this)) if aps_this else 0.0)
    cv_zs_summary = {qt: {"mean": round(float(np.mean(v)), 4),
                          "std": round(float(np.std(v)), 4),
                          "n_folds": len(v)} for qt, v in cv_zs.items() if v}
    cv_macro_mean = round(float(np.mean(fold_macro)), 4)
    cv_macro_std = round(float(np.std(fold_macro)), 4)

    out = {
        "dataset": "dynamic_earthnet", "encoder": args.encoder,
        "color_mode": args.color_mode, "approach": args.approach,
        "n_pairs": N, "n_aois": len(aois), "folds": args.folds,
        "full_corpus": {"macro_mAP": macro_full, "per_query": full},
        "kfold_zero_shot": {"macro_mAP_mean": cv_macro_mean, "macro_mAP_std": cv_macro_std,
                            "fold_macro": [round(x, 4) for x in fold_macro],
                            "per_query": cv_zs_summary},
    }

    # --- 3. leakage-free k-fold PEFT (optional) ------------------------------
    if args.peft:
        from src.train import TrainConfig, train_adapter
        key = {(p.location_id, p.t1_key, p.t2_key): i for i, p in enumerate(pairs)}
        peft_macro = []
        peft_pq = {q.text: [] for q in evaluable}
        for k in range(args.folds):
            test_idx = fold_idx(k)
            train_aois = [a for a in aois if aoi_fold[a] != k]
            ds_tr = DENNpyDataset(root=args.root, split=None,
                                  aoi_filter=train_aois, color_mode=args.color_mode)
            tr_pairs = ds_tr.list_pairs()
            tr_idx = np.array([key[(p.location_id, p.t1_key, p.t2_key)] for p in tr_pairs])
            store_tr = _sub_store(store, tr_idx)
            cfg = TrainConfig(mode="difference", epochs=args.epochs, seed=args.seed)
            adapter, _ = train_adapter(ds_tr, store_tr, enc, cfg, verbose=False)
            sub = _sub_store(store, test_idx)
            rsub = ChangeRetriever(sub, enc, feature_mode="difference")
            rsub.set_adapter(adapter, feature_mode="difference")
            aps_this = []
            for q in evaluable:
                rel = rel_all[q.text][test_idx]
                if rel.sum() == 0:
                    continue
                sc = rsub.score_all(q.text, approach="peft")
                ap_k = _average_precision(rel[np.argsort(-sc)])
                peft_pq[q.text].append(ap_k)
                aps_this.append(ap_k)
            peft_macro.append(float(np.mean(aps_this)) if aps_this else 0.0)
            print(f"  [peft fold {k}] macro mAP={peft_macro[-1]:.4f} (train {len(tr_pairs)} pairs)")
        out["kfold_peft"] = {
            "macro_mAP_mean": round(float(np.mean(peft_macro)), 4),
            "macro_mAP_std": round(float(np.std(peft_macro)), 4),
            "fold_macro": [round(x, 4) for x in peft_macro],
            "per_query": {qt: {"mean": round(float(np.mean(v)), 4),
                               "std": round(float(np.std(v)), 4), "n_folds": len(v)}
                          for qt, v in peft_pq.items() if v},
        }

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    op = Path(args.results_dir) / f"cv_eval__{args.encoder}__{args.color_mode}__{args.approach}.json"
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nfull-corpus macro mAP = {macro_full} ({len(full)} queries)")
    print(f"k-fold zero_shot macro mAP = {cv_macro_mean} ± {cv_macro_std} (n={args.folds} folds)")
    if args.peft:
        print(f"k-fold PEFT macro mAP = {out['kfold_peft']['macro_mAP_mean']} "
              f"± {out['kfold_peft']['macro_mAP_std']}")
    print(f"wrote {op}")
    for r in full:
        print(f"  {r['ap']:.3f}  CI{r['ci95']}  p={r['perm_p']:.3f}  n={r['n_relevant']:3d}  {r['query'][:46]}")


if __name__ == "__main__":
    main()
