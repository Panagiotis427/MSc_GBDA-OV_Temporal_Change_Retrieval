"""
SECOND-CC open-vocabulary change-retrieval benchmark.

Encodes (or loads cached) per-pair embeddings for a SECOND-CC split and scores the
caption-grounded queries (``src/queries/second_cc.py``) under the naive and
zero-shot approaches, for each frozen encoder. SECOND-CC is the open-vocabulary
breadth test the other datasets lack (six land-cover classes + road), so per-query
AP exposes how recovery varies across change types. Writes one JSON per encoder to
``results/`` with per-query AP, like the LEVIR-CC / DEN / QFabric results. The
shared embed->score->JSON flow lives in ``scripts._caption_benchmark``.

Run::

    python -m scripts.benchmark_second_cc --root data/_second_cc/extracted/SECOND-CC-AUG
"""
from __future__ import annotations

from scripts._caption_benchmark import run_caption_benchmark

_DEFAULT_ROOT = "data/_second_cc/extracted/SECOND-CC-AUG"
_ROOT_HELP = "SECOND-CC dir (SECOND-CC-AUG.json + {split}/rgb/{A,B})"


def main() -> None:
    run_caption_benchmark(
        "second_cc", display_name="SECOND-CC",
        default_root=_DEFAULT_ROOT, root_help=_ROOT_HELP,
    )


if __name__ == "__main__":
    main()
