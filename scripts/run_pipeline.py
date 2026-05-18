"""
One-command reproducibility runner for the change-retrieval pipeline:

    data -> per-pair embeddings -> zero-shot benchmark -> PEFT train ->
    PEFT benchmark -> printed comparison table

Every heavy step is cached, so re-runs are fast. Use ``--encoder`` to repeat
for clip_vitl14 / georsclip / remoteclip and compare.

Examples
--------
Smoke (synthetic fixture, no downloads, ~seconds on CPU):
    python -m scripts.run_pipeline --root tests/fixtures/den_tiny --epochs 30

Real DEN subset, train-on-train / eval-on-test (default):
    python -m scripts.run_pipeline --root data/DynamicEarthNet \\
        --encoder clip_vitl14 --epochs 40

All splits evaluated independently:
    python -m scripts.run_pipeline --root data/DynamicEarthNet \\
        --encoder clip_vitl14 --eval-splits train val test --epochs 40

NRG false-colour (NIR-Red-Green) instead of RGB:
    python -m scripts.run_pipeline --root data/DynamicEarthNet \\
        --encoder clip_vitl14 --color-mode nrg --epochs 40
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.datasets.registry import build_dataset
from src.embeddings import load_or_compute
from src.encoders import get_encoder
from src.retrieval import ChangeRetriever
from src.benchmark import run_benchmark
from src.train import TrainConfig, train_adapter
from src.model import save_adapter


def _build_ds(args, split, **extra):
    return build_dataset(
        args.dataset,
        root=args.root,
        pairing=args.pairing,
        split=split,
        color_mode=args.color_mode,
        **extra,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproducible change-retrieval pipeline")
    ap.add_argument("--dataset", default="dynamic_earthnet")
    ap.add_argument("--root", default="data/DynamicEarthNet")
    ap.add_argument("--pairing", default="bimonthly",
                    choices=["bimonthly", "monthly", "seasonal-quartet"])
    ap.add_argument("--encoder", default="clip_vitl14",
                    choices=["clip_vitl14", "georsclip", "remoteclip"])
    ap.add_argument("--mode", default="difference",
                    choices=["difference", "concatenate"])
    ap.add_argument("--color-mode", default="rgb",
                    choices=["rgb", "nrg", "ndvi"],
                    help="Image colour mode: rgb (default) | nrg (NIR-R-G) | ndvi")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--train-split", default="train",
                    help="Split used to train the PEFT adapter (default: train)")
    ap.add_argument("--eval-splits", nargs="+", default=["test"],
                    help="Splits to evaluate on (default: test). Pass 'all' for all splits.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    enc = get_encoder(args.encoder)
    color_tag = f"_{args.color_mode}" if args.color_mode != "rgb" else ""

    # ------------------------------------------------------------------
    # Training split: compute embeddings + train adapter
    # ------------------------------------------------------------------
    ds_train = _build_ds(args, split=args.train_split)
    store_train = load_or_compute(
        ds_train, enc,
        cache_dir=args.cache_dir,
        cache_tag=f"{args.train_split}{color_tag}",
    )

    retr_train = ChangeRetriever(store_train, enc, feature_mode=args.mode)

    print(f"\n=== TRAINING SPLIT: {args.train_split.upper()} "
          f"({len(store_train)} pairs) ===")
    train_summary: dict[str, float] = {}
    for appr in ("naive", "zero_shot"):
        rep = run_benchmark(ds_train, retr_train, approach=appr)
        print(rep.to_table())
        train_summary[appr] = rep.mAP

    adapter = None
    if not args.skip_train:
        print(f"\nTraining PEFT adapter ({args.mode}, {args.epochs} epochs)...")
        cfg = TrainConfig(mode=args.mode, epochs=args.epochs, seed=args.seed)
        adapter, _ = train_adapter(ds_train, store_train, enc, cfg)
        adapter_path = (
            f"models/{ds_train.name}__{enc.name}{color_tag}__adapter.pt"
        )
        save_adapter(adapter_path, adapter, {
            "input_dim": adapter.input_dim,
            "output_dim": adapter.output_dim,
            "hidden_dims": list(cfg.hidden_dims),
            "dropout_rate": cfg.dropout,
            "feature_mode": args.mode,
            "encoder_name": enc.name,
            "dataset_name": ds_train.name,
            "train_split": args.train_split,
            "color_mode": args.color_mode,
        })
        print(f"Saved adapter -> {adapter_path}")
        retr_train.set_adapter(adapter, feature_mode=args.mode)
        rep = run_benchmark(ds_train, retr_train, approach="peft")
        print(rep.to_table())
        train_summary["peft"] = rep.mAP

    # ------------------------------------------------------------------
    # Evaluation splits
    # ------------------------------------------------------------------
    eval_splits = args.eval_splits
    if eval_splits == ["all"]:
        eval_splits = ["train", "val", "test"]

    eval_results: dict[str, dict[str, float]] = {}

    for esplit in eval_splits:
        if esplit == args.train_split and len(eval_splits) == 1:
            # Already benchmarked above; skip redundant re-run
            eval_results[esplit] = train_summary
            continue

        print(f"\n=== EVAL SPLIT: {esplit.upper()} ===")
        ds_eval = _build_ds(args, split=esplit if esplit != "all" else None)
        store_eval = load_or_compute(
            ds_eval, enc,
            cache_dir=args.cache_dir,
            cache_tag=f"{esplit}{color_tag}",
        )
        retr_eval = ChangeRetriever(store_eval, enc, feature_mode=args.mode)

        split_summary: dict[str, float] = {}
        for appr in ("naive", "zero_shot"):
            rep = run_benchmark(ds_eval, retr_eval, approach=appr)
            print(rep.to_table())
            split_summary[appr] = rep.mAP

        if adapter is not None:
            retr_eval.set_adapter(adapter, feature_mode=args.mode)
            rep = run_benchmark(ds_eval, retr_eval, approach="peft")
            print(rep.to_table())
            split_summary["peft"] = rep.mAP

        eval_results[esplit] = split_summary

    # ------------------------------------------------------------------
    # Final comparison table
    # ------------------------------------------------------------------
    print("\n============= FINAL COMPARISON (mAP) =============")
    print(f"encoder={enc.name}  mode={args.mode}  "
          f"color={args.color_mode}  pairing={args.pairing}")

    approaches = sorted({k for v in eval_results.values() for k in v})
    header = f"{'split':10s}" + "".join(f"  {a:10s}" for a in approaches)
    print(header)
    print("-" * len(header))
    for sp, res in eval_results.items():
        row = f"{sp:10s}" + "".join(
            f"  {res.get(a, float('nan')):10.4f}" for a in approaches
        )
        print(row)
    print("===================================================")


if __name__ == "__main__":
    main()
