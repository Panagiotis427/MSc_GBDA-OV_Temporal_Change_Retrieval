"""
Per-class / confusion error analysis for change retrieval.

Where ``src.benchmark`` reports aggregate Recall@K / mAP, this module asks
*what* a query retrieves wrongly: for each natural-language query it bins the
top-K retrieved pairs by their **actual** label transition (e.g.
``forest_and_other_vegetation->soil``, ``seasonal:snow_and_ice``, ``stable``)
and tallies precision/recall plus the false-positive transition mix. The result
is a ``[query x actual-transition]`` confusion matrix — the artefact behind the
graded seasonal-vs-permanent error analysis (it makes visible when seasonal
transitions leak into permanent-change queries).

Reuses the exact scoring path of ``run_benchmark`` (``ChangeRetriever.score_all``
+ query relevance predicates), so numbers are consistent with the benchmark.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.benchmark import Query, _is_seasonal, _t1, _t2
from src.datasets.base import PairLabel
from src.retrieval import ChangeRetriever
from src.stats import rank_order


def _actual_transition(lb: Optional[PairLabel]) -> str:
    """Coarse string label for a pair's true transition (the confusion columns)."""
    if lb is None:
        return "unlabeled"
    if _is_seasonal(lb):
        snow = "snow_and_ice"
        other = _t2(lb) if _t1(lb) == snow else _t1(lb)
        return f"seasonal:{other}" if other and other != snow else "seasonal:snow_and_ice"
    if lb.stable or _t1(lb) == _t2(lb):
        return "stable"
    return f"{_t1(lb)}->{_t2(lb)}"


@dataclass
class PerQueryErrors:
    text: str
    category: str
    n_relevant: int
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    fp_transitions: Dict[str, int]  # actual transition -> count among non-relevant top-K
    fp_seasonal: int                # non-relevant top-K that are seasonal

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "category": self.category,
            "n_relevant": int(self.n_relevant),
            "precision_at_k": {str(k): float(v) for k, v in self.precision_at_k.items()},
            "recall_at_k": {str(k): float(v) for k, v in self.recall_at_k.items()},
            "fp_transitions": dict(self.fp_transitions),
            "fp_seasonal": int(self.fp_seasonal),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PerQueryErrors":
        return cls(
            text=d["text"], category=d["category"], n_relevant=int(d["n_relevant"]),
            precision_at_k={int(k): float(v) for k, v in d["precision_at_k"].items()},
            recall_at_k={int(k): float(v) for k, v in d["recall_at_k"].items()},
            fp_transitions={k: int(v) for k, v in d["fp_transitions"].items()},
            fp_seasonal=int(d["fp_seasonal"]),
        )


@dataclass
class ConfusionReport:
    encoder: str
    approach: str
    dataset: str
    split: Optional[str]
    conf_k: int
    labels: List[str]           # actual-transition columns
    matrix: np.ndarray          # [n_query, n_labels] counts over top-conf_k
    per_query: List[PerQueryErrors] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "encoder": self.encoder, "approach": self.approach,
            "dataset": self.dataset, "split": self.split, "conf_k": self.conf_k,
            "labels": list(self.labels),
            "query_texts": [q.text for q in self.per_query],
            "matrix": self.matrix.astype(int).tolist(),
            "per_query": [q.to_dict() for q in self.per_query],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ConfusionReport":
        return cls(
            encoder=d["encoder"], approach=d["approach"], dataset=d["dataset"],
            split=d.get("split"), conf_k=int(d["conf_k"]),
            labels=list(d["labels"]),
            matrix=np.array(d["matrix"], dtype=int),
            per_query=[PerQueryErrors.from_dict(q) for q in d["per_query"]],
        )


def build_confusion(
    dataset,
    retriever: ChangeRetriever,
    approach: str = "zero_shot",
    queries: Optional[List[Query]] = None,
    k_values=(1, 3, 5, 10),
    conf_k: int = 10,
    max_cols: int = 12,
    split: Optional[str] = None,
) -> ConfusionReport:
    if queries is None:
        from src.queries import get_queries
        queries = get_queries(dataset.name)
    pairs = retriever.store.pairs
    labels = [dataset.get_pair_label(p) for p in pairs]
    actual = [_actual_transition(lb) for lb in labels]

    rows: List[Counter] = []
    per_query: List[PerQueryErrors] = []
    col_counts: Counter = Counter()

    for q in queries:
        rel = np.array([bool(lb is not None and q.predicate(lb)) for lb in labels])
        if rel.sum() == 0:
            continue  # not evaluable in this corpus
        scores = retriever.score_all(q.text, approach=approach)
        order = rank_order(scores, rel)

        prec, rec = {}, {}
        for kk in k_values:
            topk = order[:kk]
            prec[kk] = float(rel[topk].sum() / kk)
            rec[kk] = float(rel[topk].sum() / rel.sum())

        top = order[:conf_k]
        row_counter = Counter(actual[i] for i in top)
        col_counts.update(row_counter)
        fp_trans = Counter(actual[i] for i in top if not rel[i])
        fp_seasonal = int(sum(1 for i in top if (not rel[i]) and _is_seasonal(labels[i])))

        rows.append(row_counter)
        per_query.append(PerQueryErrors(
            text=q.text, category=q.category, n_relevant=int(rel.sum()),
            precision_at_k=prec, recall_at_k=rec,
            fp_transitions=dict(fp_trans), fp_seasonal=fp_seasonal,
        ))

    # Columns = most frequent actual transitions; rare ones folded into "other".
    common = [t for t, _ in col_counts.most_common(max_cols)]
    use_other = len(col_counts) > len(common)
    cols = common + (["other"] if use_other else [])
    col_idx = {t: i for i, t in enumerate(common)}

    M = np.zeros((len(rows), len(cols)), dtype=int)
    for i, rc in enumerate(rows):
        for t, c in rc.items():
            if t in col_idx:
                M[i, col_idx[t]] += c
            elif use_other:
                M[i, len(common)] += c

    return ConfusionReport(
        encoder=retriever.encoder.name, approach=approach, dataset=dataset.name,
        split=split, conf_k=conf_k, labels=cols, matrix=M, per_query=per_query,
    )


def confusion_to_csv(report: ConfusionReport, path) -> None:
    import csv
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query"] + report.labels)
        for q, row in zip(report.per_query, report.matrix):
            w.writerow([q.text] + list(row))
