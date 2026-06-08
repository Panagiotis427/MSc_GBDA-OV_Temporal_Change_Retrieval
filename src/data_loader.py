"""
Data loader for multitemporal satellite imagery embeddings.

This module provides utilities to load pre-computed QFabric or Dynamic EarthNet features,
parsing metadata to extract geographic coordinates and timestamps. It creates PyTorch
datasets of paired (embedding_T1, embedding_T2) vectors for change detection tasks.

.. deprecated::
    For new code, use the ``src.datasets`` subpackage instead:

        from src.datasets import get_dataset
        ds = get_dataset("qfabric", parquet_paths=[...], cache_path=...)

    Only the QFabric parquet→embedding helpers (``load_parquet_as_embeddings``,
    ``parse_date_from_filename``) remain here as a backwards-compatibility facade
    used by ``src.datasets.qfabric``. The old ``TemporalEmbeddingDataset`` /
    ``load_qfabric_features`` (which were non-functional) have been removed in
    favour of ``src.datasets.QFabricDataset``.
"""
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import pandas as pd
import numpy as np


LEGACY_CACHE_NAME = "clip_embeddings.npz"


def cache_path(data_dir: Union[str, Path], dataset: str, encoder: str) -> Path:
    """Canonical embedding-cache path for a (dataset, encoder) combination.

    Example: ``cache_path("data", "qfabric", "clip_vitl14")``
    → ``data/cache/qfabric__clip_vitl14__embeddings.npz``
    """
    return Path(data_dir) / "cache" / f"{dataset}__{encoder}__embeddings.npz"


def migrate_legacy_cache(data_dir: Union[str, Path]) -> Optional[Path]:
    """Rename ``data/clip_embeddings.npz`` to the new cache filename if needed.

    Idempotent — does nothing if either (a) the legacy file is absent, or
    (b) the new file already exists. Returns the new path when a migration
    actually happened, else None.
    """
    data_dir = Path(data_dir)
    legacy = data_dir / LEGACY_CACHE_NAME
    new = cache_path(data_dir, dataset="qfabric", encoder="clip_vitl14")
    if not legacy.exists() or new.exists():
        return None
    new.parent.mkdir(parents=True, exist_ok=True)
    legacy.rename(new)
    print(f"Migrated legacy cache: {legacy} -> {new}")
    return new


def parse_date_from_filename(filename: str) -> pd.Timestamp:
    """Parse date from QFabric filename like '69.d1.02022015_0_1024.tif'."""
    try:
        date_str = filename.split('.')[2].split('_')[0]  # '02022015'
        return pd.Timestamp(f"{date_str[4:8]}-{date_str[:2]}-{date_str[2:4]}")
    except Exception:
        return pd.Timestamp('2015-01-01')


def load_parquet_as_embeddings(
    parquet_paths,
    clip_model,
    processor,
    device,
    cache_path: Optional[str] = None
) -> Tuple[dict, pd.DataFrame, dict]:
    """
    Load one or more QFabric parquet shards, encode images through CLIP.

    Args:
        parquet_paths: A single path string or a list of path strings.
        clip_model: HuggingFace CLIP model (must have get_image_features).
        processor: CLIPProcessor for image preprocessing.
        device: torch.device for inference.
        cache_path: Optional .npz path for caching computed embeddings.
                    If it exists, CLIP encoding is skipped on subsequent runs.

    Returns:
        embedding_lookup: {loc_id: np.ndarray shape [n_timepoints, 768]}
        metadata_df: DataFrame with columns ['location', 'timestamp', 'timepoint_idx']
        image_lookup: {loc_id: list of PIL Images (one per timepoint)}
    """
    import io
    import torch
    from PIL import Image as PILImage

    if isinstance(parquet_paths, str):
        parquet_paths = [parquet_paths]

    # Build image_lookup and metadata from all shards (fast — just PNG decompression)
    image_lookup: dict = {}
    metadata_rows = []

    for parquet_path in parquet_paths:
        shard_id = os.path.splitext(os.path.basename(parquet_path))[0]  # e.g. 'train-00000-of-00597'
        df = pd.read_parquet(parquet_path)
        img_cols = [c for c in df.columns if c.endswith('_image') and not c.endswith('_name')]

        print(f"Reading images from {os.path.basename(parquet_path)} ({len(df)} locations)...")
        for row_idx, row in df.iterrows():
            loc_id = f"{shard_id}_r{row_idx:04d}"
            images = []
            for t_idx, img_col in enumerate(img_cols):
                # Match the name column to its image column by name
                # (``t1_image`` -> ``t1_image_name``) instead of zipping two
                # independently-filtered lists (order-fragile).
                name_col = img_col + "_name"
                img = PILImage.open(io.BytesIO(row[img_col]['bytes'])).convert('RGB')
                images.append(img)
                timestamp = (parse_date_from_filename(row[name_col])
                             if name_col in df.columns else pd.Timestamp('2015-01-01'))
                metadata_rows.append({'location': loc_id, 'timestamp': timestamp, 'timepoint_idx': t_idx})
            image_lookup[loc_id] = images

    metadata_df = pd.DataFrame(metadata_rows)

    # Load embeddings from cache if available
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        data = np.load(cache_path)
        embedding_lookup = {k: data[k] for k in data.files}
        # If new shards were added, encode only the missing locations
        missing = [loc for loc in image_lookup if loc not in embedding_lookup]
        if missing:
            print(f"Cache found but {len(missing)} new locations need encoding...")
            embedding_lookup = _encode_locations(missing, image_lookup, clip_model, processor, device, embedding_lookup)
            np.savez(cache_path, **embedding_lookup)
        else:
            print(f"Loaded {len(embedding_lookup)} locations from cache.")
        return embedding_lookup, metadata_df, image_lookup

    # Encode all locations through CLIP
    embedding_lookup = {}
    embedding_lookup = _encode_locations(list(image_lookup.keys()), image_lookup, clip_model, processor, device, embedding_lookup)

    if cache_path:
        print(f"Saving embeddings cache to {cache_path}...")
        np.savez(cache_path, **embedding_lookup)

    print(f"Done. {len(embedding_lookup)} locations encoded.")
    return embedding_lookup, metadata_df, image_lookup


def _encode_locations(loc_ids, image_lookup, clip_model, processor, device, existing=None):
    """Encode a list of locations through CLIP and merge into existing dict."""
    import torch
    result = dict(existing) if existing else {}
    total = len(loc_ids)
    print(f"Encoding {total} locations through CLIP...")
    for i, loc_id in enumerate(loc_ids):
        embeddings = []
        for img in image_lookup[loc_id]:
            pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                out = clip_model.get_image_features(pixel_values=pixel_values)
                emb = out if isinstance(out, torch.Tensor) else out.pooler_output
            embeddings.append(emb.squeeze(0).cpu().numpy())
        result[loc_id] = np.array(embeddings, dtype=np.float32)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {i + 1}/{total} locations encoded...")
    return result
