"""
Build the QFabric per-crop change-type label index from TEOChatlas RQA2.

Downloads (or reads) ``eval/QFabric_RQA2.json`` from ``jirvin16/TEOChatlas`` and
emits ``data/QFabric/qfabric_teo_labels.json`` mapping each crop key
(``<loc>_<xoff>_<yoff>``) to its real change type (one of the QFabric 6:
residential / commercial / industrial / road / demolition / mega_projects).

These are the authentic QFabric labels (the RQA2 questions' answers), joined to
the crop images by the shared filename scheme — no manual rating, no spatial
join. Consumed by ``src.datasets.qfabric_teo.TEOChatlasQFabricDataset``.

Run::

    python -m scripts.build_qfabric_labels --out data/QFabric/qfabric_teo_labels.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.datasets.qfabric_teo import parse_crop

_REPO = "jirvin16/TEOChatlas"
_RQA2 = "eval/QFabric_RQA2.json"
_NORM = {"Residential": "residential", "Commercial": "commercial",
         "Industrial": "industrial", "Road": "road",
         "Demolition": "demolition", "Mega Projects": "mega_projects"}


def _gpt_answer(conversations) -> str | None:
    for c in conversations:
        if c.get("from") == "gpt":
            return (c.get("value") or "").strip()
    return None


def build(rqa2_path: str, out_path: str) -> dict:
    data = json.load(open(rqa2_path, encoding="utf-8"))
    # A crop can have several RQA2 records (multiple polygons / overlapping
    # answers). Resolve by MAJORITY VOTE over all answers for that crop_key —
    # not last-write-wins (which mislabels ~16% of crops with conflicts).
    votes: dict[str, Counter] = defaultdict(Counter)
    for r in data:
        vids = r.get("video") or []
        if not vids:
            continue
        ans = _gpt_answer(r.get("conversations", []))
        matched = [v for k, v in _NORM.items() if ans and k.lower() in ans.lower()]
        if not matched:
            continue
        parsed = parse_crop(vids[0])
        if parsed is None:
            continue
        votes[parsed[0]][matched[0]] += 1
    # most_common is deterministic for ties (insertion order); fine for labels.
    crop_type = {ck: c.most_common(1)[0][0] for ck, c in votes.items()}
    n_conflict = sum(1 for c in votes.values() if len(c) > 1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(crop_type, open(out_path, "w"), indent=0)
    print(f"Wrote {len(crop_type)} crop labels -> {out_path} "
          f"({n_conflict} crops had conflicting answers, resolved by majority)")
    print("change-type distribution:", dict(Counter(crop_type.values())))
    return crop_type


def main() -> None:
    ap = argparse.ArgumentParser(description="Build QFabric crop->change-type labels")
    ap.add_argument("--rqa2", default=None,
                    help="Path to QFabric_RQA2.json (downloaded from TEOChatlas if omitted)")
    ap.add_argument("--out", default="data/QFabric/qfabric_teo_labels.json")
    args = ap.parse_args()

    rqa2 = args.rqa2
    if rqa2 is None:
        from huggingface_hub import hf_hub_download
        rqa2 = hf_hub_download(_REPO, _RQA2, repo_type="dataset",
                               local_dir="data/QFabric/_teochatlas")
        print(f"Downloaded {rqa2}")
    build(rqa2, args.out)


if __name__ == "__main__":
    main()
