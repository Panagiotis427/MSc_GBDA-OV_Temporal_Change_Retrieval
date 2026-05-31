"""
Download a SUBSET of the QFabric multi-temporal image dataset from HuggingFace.

Source: ``EVER-Z/QFabric_mt_images_1024`` (parquet shards under ``data/`` in the
repo; each shard ~tens of MB, ~87 locations x 5 timepoints). Only a handful of
shards are pulled by default to stay well under the supervisor's <10-20 GB
constraint — enough to prove the dataset-agnostic pipeline end-to-end.

NOTE: these shards are **images only** — there are no change/status label
columns (see docs/EXTENSIONS.md). ``QFabricDataset.get_pair_label`` therefore
returns ``None`` and the label-grounded benchmark is not run for QFabric; the
dataset is exercised qualitatively (zero-shot retrieval in the Gradio app).

Usage::

    python -m scripts.download_qfabric --dest data/QFabric --n-shards 8
"""
import argparse
from pathlib import Path

_HF_REPO = "EVER-Z/QFabric_mt_images_1024"
_HF_DIR = "data"               # parquet shards live under data/ in the repo
_N_SHARDS_DEFAULT = 8          # subset; ~700 locations, well under 20 GB


def _list_shard_files(repo: str) -> list[str]:
    from huggingface_hub import list_repo_files
    files = list_repo_files(repo, repo_type="dataset")
    shards = sorted(f for f in files
                    if f.startswith(f"{_HF_DIR}/") and f.endswith(".parquet"))
    if not shards:
        raise RuntimeError(f"No parquet shards found under {_HF_DIR}/ in {repo}.")
    return shards


def download_shards(repo: str, n_shards: int, dest_dir: Path) -> list[Path]:
    """Download the first ``n_shards`` parquet shards (idempotent per file)."""
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    shards = _list_shard_files(repo)[:n_shards]
    out: list[Path] = []
    for i, rel in enumerate(shards):
        local = dest_dir / Path(rel).name
        if local.exists():
            print(f"  [{i+1}/{len(shards)}] present: {local.name}")
        else:
            print(f"  [{i+1}/{len(shards)}] downloading {rel} ...")
            p = hf_hub_download(repo, rel, repo_type="dataset",
                                local_dir=str(dest_dir))
            # hf_hub_download with subdir keeps the data/ prefix; flatten it.
            p = Path(p)
            if p != local:
                p.replace(local)
        out.append(local)
    # Clean any empty data/ subdir left by hf_hub_download.
    sub = dest_dir / _HF_DIR
    if sub.is_dir() and not any(sub.iterdir()):
        sub.rmdir()
    return out


def verify_layout(root: Path, min_shards: int = 1) -> None:
    import pandas as pd
    shards = sorted(root.glob("*.parquet"))
    if len(shards) < min_shards:
        raise RuntimeError(f"Found {len(shards)} parquet shard(s) under {root} "
                           f"(need >= {min_shards}).")
    df = pd.read_parquet(shards[0], columns=None)
    img_cols = [c for c in df.columns if c.endswith("_image") and not c.endswith("_name")]
    name_cols = [c for c in df.columns if c.endswith("_image_name")]
    if len(img_cols) != 5 or len(name_cols) != 5:
        raise RuntimeError(f"Unexpected schema in {shards[0].name}: "
                           f"{len(img_cols)} image cols, {len(name_cols)} name cols (want 5/5).")
    print(f"Layout OK — {len(shards)} shard(s); first has {len(df)} locations, "
          f"{len(img_cols)} timepoints/row.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download a QFabric image subset")
    ap.add_argument("--dest", default="data/QFabric")
    ap.add_argument("--n-shards", type=int, default=_N_SHARDS_DEFAULT)
    ap.add_argument("--repo", default=_HF_REPO)
    ap.add_argument("--skip-download", action="store_true",
                    help="Only verify shards already present.")
    args = ap.parse_args()

    root = Path(args.dest)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "_done.marker"
    if marker.exists():
        print(f"QFabric already prepared at {root} (found _done.marker).")
        return

    if not args.skip_download:
        download_shards(args.repo, args.n_shards, root)
    verify_layout(root)
    marker.touch()
    print(f"\nQFabric subset ready at: {root}  (images-only; labels not included)")
    print("Try it:  python -m src.app --dataset qfabric --root data/QFabric --approach zero_shot")


if __name__ == "__main__":
    main()
