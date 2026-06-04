"""
Download and prepare the 5-AOI Dynamic EarthNet preprocessed subset.

Usage:
    python -m scripts.download_den [--dest data/DynamicEarthNet]

Steps
-----
1. gdown the archive (idempotent — skips if ``_done.marker`` exists).
2. Extract (auto-detects .tar / .zip / .tar.gz).
3. Verify ≥ 5 AOI subdirs under ``planet/``.
4. Build ``labels_index.parquet`` by iterating all candidate pairs and calling
   ``derive_pair_label``.
5. Touch ``_done.marker``.
"""
import argparse
import json
import os
import tarfile
import zipfile
from pathlib import Path

# gdown ID for the 5-AOI preprocessed DEN subset (~7 GB extracted)
_GDRIVE_ID = "1cMP57SPQWYKMy8X60iK217C28RFBkd2z"
_ARCHIVE_NAME = "den_5aoi.tar.gz"


def _gdown_download(gdrive_id: str, dest_file: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required for DEN download. Install: pip install gdown"
        ) from exc
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(f"Downloading DEN archive from Google Drive ({gdrive_id})...")
    gdown.download(url, str(dest_file), quiet=False)


def _extract(archive: Path, dest_dir: Path) -> None:
    # Sniff the real format (the gdown file is a .zip mislabelled .tar.gz),
    # so dispatch on content, not the extension.
    print(f"Extracting {archive.name} -> {dest_dir} ...")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest_dir)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as tf:   # auto-detects gz/bz2/xz/plain
            tf.extractall(dest_dir)
    else:
        raise ValueError(
            f"Unrecognised archive format for {archive} "
            "(not zip or tar). Re-download may be corrupt.")
    print("Extraction done.")


def verify_layout(root: Path, min_aois: int = 5) -> None:
    planet_dir = root / "planet"
    if not planet_dir.exists():
        raise RuntimeError(
            f"Expected 'planet/' subdir under {root}.  "
            "Check archive contents; layout may differ."
        )
    aois = [d for d in planet_dir.iterdir() if d.is_dir()]
    if len(aois) < min_aois:
        raise RuntimeError(
            f"Found only {len(aois)} AOI dirs under {planet_dir} (need >= {min_aois})."
        )
    print(f"Layout OK — {len(aois)} AOIs found: {[a.name for a in sorted(aois)]}")


def build_label_index(root: Path, pairing_strategy: str = "bimonthly") -> Path:
    """Iterate all pairs via ``DENDataset`` and write the ``labels_index.parquet``.

    Columns: ``location``, ``t1_key``, ``t2_key``, ``change_type``, ``stable``,
    ``dominant_t1_class``, ``dominant_t2_class``, and ``class_change_fraction_json``
    (JSON-encoded ``{class: {gained_fraction, lost_fraction}}`` per pair). The
    fractions let the loader serve fraction-based relevance predicates without
    re-reading rasters.
    """
    import pandas as pd
    from src.datasets.dynamic_earthnet import FRAC_JSON_COL, DENDataset

    print("Building labels_index.parquet ...")
    ds = DENDataset(root, pairing_strategy=pairing_strategy)
    rows = []
    for pair in ds.list_pairs():
        label = ds.get_pair_label(pair)
        rows.append({
            "location": pair.location_id,
            "t1_key": pair.t1_key,
            "t2_key": pair.t2_key,
            "change_type": label.change_type if label else "unknown",
            "stable": label.stable if label else True,
            "dominant_t1_class": label.dominant_t1_class if label else None,
            "dominant_t2_class": label.dominant_t2_class if label else None,
            # Persist the per-class change fractions so the index fast-path can
            # serve fraction-based relevance predicates without re-reading the
            # rasters (and consistently with the scalar fields above).
            FRAC_JSON_COL: json.dumps(
                label.class_change_mask_fraction if label else {}
            ),
        })
    df = pd.DataFrame(rows)
    out_path = root / "labels_index.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} pair labels -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download + prepare DEN 5-AOI subset")
    parser.add_argument("--dest", type=str, default="data/DynamicEarthNet",
                        help="Root directory for DEN data")
    parser.add_argument("--gdrive-id", type=str, default=_GDRIVE_ID,
                        help="Google Drive file ID for the archive")
    parser.add_argument("--strategy", type=str, default="bimonthly",
                        choices=["bimonthly", "monthly", "seasonal-quartet"],
                        help="Pair-building strategy for labels_index")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download (archive already present)")
    args = parser.parse_args()

    root = Path(args.dest)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "_done.marker"

    if marker.exists():
        print(f"DEN already prepared at {root} (found _done.marker). Nothing to do.")
        return

    raw_dir = root / "raw"
    raw_dir.mkdir(exist_ok=True)
    archive = raw_dir / _ARCHIVE_NAME

    if not args.skip_download:
        if not archive.exists():
            _gdown_download(args.gdrive_id, archive)
        else:
            print(f"Archive already present: {archive}")
    else:
        if not archive.exists():
            raise FileNotFoundError(
                f"--skip-download set but archive not found at {archive}"
            )

    from src.datasets.dynamic_earthnet_pp import resolve_pp_root

    already = resolve_pp_root(root) is not None or (root / "planet").is_dir()
    if not already:
        _extract(archive, root)

    pp = resolve_pp_root(root)
    if pp is not None:
        # Preprocessed DynNet subset: labels are self-contained .npy arrays;
        # no raster verify / parquet index needed (DENNpyDataset derives
        # labels on the fly).
        n_aoi = len(list((pp / "labels").glob("*.npy")))
        print(f"Preprocessed DEN detected at {pp} ({n_aoi} AOIs). "
              "No parquet index required.")
    else:
        verify_layout(root)
        build_label_index(root, pairing_strategy=args.strategy)

    marker.touch()
    print(f"\nDEN preparation complete. Data at: {root}")


if __name__ == "__main__":
    main()
