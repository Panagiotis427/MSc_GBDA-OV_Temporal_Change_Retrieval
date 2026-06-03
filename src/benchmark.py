"""
Label-grounded retrieval benchmark + seasonal-vs-permanent error analysis.

Replaces the old identity-diagonal ``train.evaluate_retrieval`` hack with a
real information-retrieval evaluation: a fixed natural-language query set,
each query mapped to a *relevance rule* over the dataset's ``PairLabel``s
(derived from DEN's pixel-wise LULC labels). We then rank the whole pair
corpus with a :class:`~src.retrieval.ChangeRetriever` and compute Recall@K
and mAP, plus an error report on "semantic drift" (seasonal transitions —
e.g. snow-melt — wrongly retrieved for permanent-change queries).

CLI:
    python -m src.benchmark --root data/DynamicEarthNet --encoder clip_vitl14 \
        --approach zero_shot
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from src.datasets.base import PairLabel, TemporalDataset
from src.datasets.registry import get_dataset
from src.embeddings import load_or_compute
from src.encoders import get_encoder
from src.retrieval import APPROACHES, ChangeRetriever

# Used by ``_is_seasonal`` to flag snow-involving transitions in the
# seasonal-drift report. Datasets without snow labels are unaffected (no
# pair matches, mask is all-False).
SNOW = "snow_and_ice"


def _t1(lb: PairLabel) -> Optional[str]:
    return lb.dominant_t1_class


def _t2(lb: PairLabel) -> Optional[str]:
    return lb.dominant_t2_class


@dataclass
class Query:
    """A natural-language query + a predicate deciding pair relevance.

    ``category`` is ``"permanent"`` or ``"seasonal"`` — used by the error
    analysis to measure seasonal/permanent confusion.
    """

    text: str
    category: str
    predicate: Callable[[PairLabel], bool]


def _transition(src=None, dst=None) -> Callable[[PairLabel], bool]:
    def pred(lb: PairLabel) -> bool:
        if lb is None or lb.stable:
            return False
        if _t1(lb) == _t2(lb):
            return False
        if src is not None and _t1(lb) != src:
            return False
        if dst is not None and _t2(lb) != dst:
            return False
        return True
    return pred


# --- fraction-based relevance --------------------------------------------------
# The default ``_transition`` predicate only fires when the *dominant* class of
# the whole tile flips — discarding all localised change (a 10% wetland gain on
# an agriculture-majority tile reads as "stable"). Empirically only 8.6% of DEN
# pairs flip dominant class, almost all wetland<->agriculture, which starves
# every other query of positives. These predicates instead use the per-class
# pixel-change fractions ``derive_pair_label`` already computes, so a pair is
# relevant when the target class gains/loses >= ``thresh`` of valid pixels.

def _gained(cls: str, thresh: float = 0.05) -> Callable[[PairLabel], bool]:
    def pred(lb: PairLabel) -> bool:
        if lb is None:
            return False
        return lb.class_change_mask_fraction.get(cls, {}).get("gained_fraction", 0.0) >= thresh
    return pred


def _lost(cls: str, thresh: float = 0.05) -> Callable[[PairLabel], bool]:
    def pred(lb: PairLabel) -> bool:
        if lb is None:
            return False
        return lb.class_change_mask_fraction.get(cls, {}).get("lost_fraction", 0.0) >= thresh
    return pred


# Per-dataset query sets live in ``src/queries/`` (one module per dataset).
# They self-register into the query registry and are resolved by
# ``run_benchmark`` via ``dataset.name``.


# Bump when the serialized JSON layout (``to_dict``) changes incompatibly.
SCHEMA_VERSION = 1


@dataclass
class QueryResult:
    text: str
    category: str
    n_relevant: int
    recall_at_k: Dict[int, float]
    ap: float
    seasonal_drift_at_k: Dict[int, float]  # frac of non-relevant top-K that are seasonal

    def to_dict(self) -> Dict:
        # K stored as *string* keys for JSON stability.
        return {
            "text": self.text,
            "category": self.category,
            "n_relevant": int(self.n_relevant),
            "recall_at_k": {str(k): float(v) for k, v in self.recall_at_k.items()},
            "ap": float(self.ap),
            "seasonal_drift_at_k": {str(k): float(v)
                                    for k, v in self.seasonal_drift_at_k.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "QueryResult":
        return cls(
            text=d["text"],
            category=d["category"],
            n_relevant=int(d["n_relevant"]),
            recall_at_k={int(k): float(v) for k, v in d["recall_at_k"].items()},
            ap=float(d["ap"]),
            seasonal_drift_at_k={int(k): float(v)
                                 for k, v in d["seasonal_drift_at_k"].items()},
        )


@dataclass
class BenchmarkReport:
    approach: str
    encoder: str
    dataset: str
    n_pairs: int
    per_query: List[QueryResult]

    @property
    def macro_recall(self) -> Dict[int, float]:
        ks = self.per_query[0].recall_at_k.keys() if self.per_query else []
        return {
            k: float(np.mean([q.recall_at_k[k] for q in self.per_query]))
            for k in ks
        }

    @property
    def mAP(self) -> float:
        return float(np.mean([q.ap for q in self.per_query])) if self.per_query else 0.0

    @property
    def _ks(self) -> List[int]:
        return sorted(self.per_query[0].recall_at_k) if self.per_query else []

    def macro_seasonal_drift(self) -> Dict[int, float]:
        """Mean seasonal-drift@K over *permanent* queries (mirrors ``to_table``).

        0.0 for every K when there are no permanent queries (or no snow class).
        """
        perm = [q for q in self.per_query if q.category == "permanent"]
        if not perm:
            return {k: 0.0 for k in self._ks}
        return {k: float(np.mean([q.seasonal_drift_at_k[k] for q in perm]))
                for k in self._ks}

    def to_dict(self, *, color_mode: str = "rgb", split: Optional[str] = None,
                lora: bool = False) -> Dict:
        ks = self._ks
        mr = self.macro_recall
        sd = self.macro_seasonal_drift()
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset": self.dataset,
            "encoder": self.encoder,
            "approach": self.approach,
            "split": split,
            "color_mode": color_mode,
            "lora": bool(lora),
            "n_pairs": int(self.n_pairs),
            "k_values": [int(k) for k in ks],
            "macro": {
                "mAP": self.mAP,
                "recall_at_k": {str(k): mr[k] for k in ks},
                "seasonal_drift_at_k": {str(k): sd[k] for k in ks},
            },
            "per_query": [q.to_dict() for q in self.per_query],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "BenchmarkReport":
        """Reconstruct the core report. Top-level ``split``/``color_mode``/``lora``
        metadata stays in the source dict (read directly by ``results_io``)."""
        return cls(
            approach=d["approach"],
            encoder=d["encoder"],
            dataset=d["dataset"],
            n_pairs=int(d["n_pairs"]),
            per_query=[QueryResult.from_dict(q) for q in d["per_query"]],
        )

    def to_table(self) -> str:
        ks = sorted(self.per_query[0].recall_at_k) if self.per_query else []
        head = (f"\n=== Benchmark: {self.dataset} | {self.encoder} | "
                f"approach={self.approach} | N={self.n_pairs} ===")
        cols = "  ".join(f"R@{k}" for k in ks)
        lines = [head, f"{'query':52s} {'#rel':>4s}  {cols}  {'AP':>5s}"]
        for q in self.per_query:
            r = "  ".join(f"{q.recall_at_k[k]:.2f}" for k in ks)
            lines.append(f"{q.text[:52]:52s} {q.n_relevant:4d}  {r}  {q.ap:5.3f}")
        mr = self.macro_recall
        lines.append("-" * len(lines[1]))
        lines.append(f"{'MACRO':52s} {'':>4s}  "
                     + "  ".join(f"{mr[k]:.2f}" for k in ks)
                     + f"  {self.mAP:5.3f}  (mAP)")
        # seasonal drift summary over permanent queries
        perm = [q for q in self.per_query if q.category == "permanent"]
        if perm:
            sd = {k: float(np.mean([q.seasonal_drift_at_k[k] for q in perm]))
                  for k in ks}
            lines.append("seasonal drift @K (permanent queries, lower=better): "
                         + "  ".join(f"R@{k}={sd[k]:.2f}" for k in ks))
        return "\n".join(lines)


def _average_precision(rel: np.ndarray) -> float:
    """rel: boolean array in ranked order. Standard AP."""
    n_rel = int(rel.sum())
    if n_rel == 0:
        return 0.0
    hits = np.cumsum(rel)
    ranks = np.arange(1, len(rel) + 1)
    precision_at_hit = (hits / ranks)[rel]
    return float(precision_at_hit.sum() / n_rel)


def _is_seasonal(lb: Optional[PairLabel]) -> bool:
    if lb is None:
        return False
    return SNOW in (lb.dominant_t1_class, lb.dominant_t2_class)


def run_benchmark(
    dataset: TemporalDataset,
    retriever: ChangeRetriever,
    approach: str = "zero_shot",
    queries: Optional[List[Query]] = None,
    k_values=(1, 3, 5, 10),
) -> BenchmarkReport:
    if queries is None:
        from src.queries import get_queries
        queries = get_queries(dataset.name)
        if not queries:
            raise ValueError(
                f"No query set registered for dataset '{dataset.name}'. "
                "Add a module under src/queries/ that calls register_queries(...)"
            )
    pairs = retriever.store.pairs
    labels = [dataset.get_pair_label(p) for p in pairs]
    seasonal_mask = np.array([_is_seasonal(lb) for lb in labels])

    per_query: List[QueryResult] = []
    for q in queries:
        rel = np.array([bool(lb is not None and q.predicate(lb)) for lb in labels])
        if rel.sum() == 0:
            continue  # query has no positives in this corpus — not evaluable
        scores = retriever.score_all(q.text, approach=approach)
        order = np.argsort(-scores)
        rel_ranked = rel[order]
        seas_ranked = seasonal_mask[order]

        recall, drift = {}, {}
        for k in k_values:
            topk = order[:k]
            recall[k] = float(rel[topk].sum() / rel.sum())
            non_rel = ~rel_ranked[:k]
            drift[k] = float(seas_ranked[:k][non_rel].mean()) if non_rel.any() else 0.0
        per_query.append(QueryResult(
            text=q.text, category=q.category, n_relevant=int(rel.sum()),
            recall_at_k=recall, ap=_average_precision(rel_ranked),
            seasonal_drift_at_k=drift,
        ))

    return BenchmarkReport(
        approach=approach, encoder=retriever.encoder.name,
        dataset=dataset.name, n_pairs=len(pairs), per_query=per_query,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Label-grounded change-retrieval benchmark")
    ap.add_argument("--dataset", default="dynamic_earthnet")
    ap.add_argument("--root", default="data/DynamicEarthNet")
    ap.add_argument("--pairing", default="bimonthly",
                    choices=["bimonthly", "monthly", "seasonal-quartet"])
    ap.add_argument("--encoder", default="clip_vitl14")
    ap.add_argument("--approach", default="zero_shot",
                    choices=list(APPROACHES) + ["all"])
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--split", default="test",
                    help="DEN preprocessed split: train|val|test|all")
    args = ap.parse_args()

    from src.datasets.registry import build_dataset
    ds = build_dataset(args.dataset, root=args.root,
                       pairing=args.pairing, split=args.split)
    enc = get_encoder(args.encoder)
    store = load_or_compute(ds, enc, cache_dir=args.cache_dir)
    retriever = ChangeRetriever(store, enc)

    approaches = ["naive", "zero_shot"] if args.approach == "all" else [args.approach]
    for appr in approaches:
        if appr == "peft":
            print("PEFT requested but no adapter wired in CLI; skip (use src.train).")
            continue
        report = run_benchmark(ds, retriever, approach=appr)
        print(report.to_table())


if __name__ == "__main__":
    main()
