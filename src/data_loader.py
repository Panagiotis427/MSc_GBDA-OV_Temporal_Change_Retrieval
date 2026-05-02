"""
Data loader for multitemporal satellite imagery embeddings.

This module provides utilities to load pre-computed QFabric or Dynamic EarthNet features,
parsing metadata to extract geographic coordinates and timestamps. It creates PyTorch
datasets of paired (embedding_T1, embedding_T2) vectors for change detection tasks.
"""
import os
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class TemporalEmbeddingDataset(Dataset):
    """PyTorch dataset for paired temporal embeddings."""

    def __init__(
        self,
        embedding_file: str,
        metadata_file: Optional[str] = None,
        transform=None
    ):
        """
        Initialize the temporal embedding dataset.

        Args:
            embedding_file (str): Path to numpy file containing embeddings.
                Shape should be (N, embedding_dim) where N is total number of observations.
            metadata_file (str, optional): Path to CSV with metadata columns including
                'location', 'timestamp'. If None, assumes filenames encode location info.
            transform: Optional spatial transforms to apply to images.

        Attributes:
            embeddings (np.ndarray): Loaded embeddings array.
            metadata (pd.DataFrame or dict): Parsed metadata mapping locations -> timestamps.
            embedding_dim (int): Dimensionality of each embedding vector.
        """
        self.embedding_file = embedding_file
        self.metadata_file = metadata_file
        self.transform = transform

        # Load embeddings
        if embedding_file.endswith('.npz'):
            data = np.load(embedding_file)
            self.embeddings = data['embeddings']
        else:
            self.embeddings = np.load(embedding_file)

        assert len(self.embeddings.shape) == 2, "Embeddings must be 2D: (N, dim)"
        self.embedding_dim = self.embeddings.shape[1]

        # Parse metadata
        if self.metadata_file is not None:
            df = pd.read_csv(metadata_file)
            required_cols = ['location', 'timestamp']
            missing = set(required_cols) - set(df.columns)
            if missing:
                raise ValueError(f"Metadata missing columns: {missing}")
            self.metadata = df
        else:
            # Fallback: extract location from filename pattern like "tile_123.npz"
            self._extract_location_from_filename()

    def _extract_location_from_filename(self):
        """Extract location identifier from embedding filename."""
        base = os.path.basename(self.embedding_file)
        # Assumes format: tile_<location_id>.npz or <location_id>_embeddings.npz
        self.location_id = base.replace('.npz', '').split('_')[-1] if '_' in base else base

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a paired temporal embedding sample.

        Args:
            idx (int): Index of the location in self.metadata. Assumes metadata is sorted
                by timestamp and contains consecutive pairs.

        Returns:
            tuple: (embedding_T1, embedding_T2) as PyTorch tensors.
                Each tensor has shape (embedding_dim,).
        """
        assert idx < len(self), f"Index {idx} out of range [0, {len(self)})"

        # Get embeddings for this location pair
        loc = self.metadata.iloc[idx]['location']
        t1_mask = self.metadata['location'] == loc & (self.metadata['timestamp'].rank() == 2*n)
        t2_mask = self.metadata['location'] == loc & (self.metadata['timestamp'].rank() == 2*n + 1)

        if not t1_mask.any() or not t2_mask.any():
            raise IndexError(f"Cannot find consecutive time pairs for location {loc}")

        emb_t1 = self.embeddings[t1_mask]
        emb_t2 = self.embeddings[t2_mask]

        assert len(emb_t1) == 1 and len(emb_t2) == 1, "Expected exactly one observation per timestamp"

        return torch.from_numpy(emb_t1).float(), torch.from_numpy(emb_t2).float()


def load_qfabric_features(
    base_path: str,
    metadata_file: Optional[str] = None
) -> TemporalEmbeddingDataset:
    """
    Load pre-computed QFabric foundation model embeddings.

    Args:
        base_path (str): Base directory containing tile-specific .npz files.
        metadata_file (str, optional): CSV with columns ['location', 'timestamp']. Defaults to None.

    Returns:
        TemporalEmbeddingDataset: Dataset ready for training loop.
    """
    # QFabric stores embeddings per tile in format: base_path/tile_<id>.npz
    all_embeddings = {}
    npz_files = sorted([f for f in os.listdir(base_path) if f.endswith('.npz')])

    for fname in npz_files:
        path = os.path.join(base_path, fname)
        data = np.load(path)
        loc_id = fname.replace('.npz', '')
        all_embeddings[loc_id] = data['embeddings'] if 'embeddings' in data else data[0]

    # Build metadata mapping
    location_to_timestamps = {}
    for loc, emb in all_embeddings.items():
        if not isinstance(emb, np.ndarray):
            continue
        timestamps = []
        for i in range(len(emb)):
            timestamp = pd.Timestamp('2023-01-01') + pd.Timedelta(days=i)  # Placeholder - replace with actual
            location_to_timestamps[loc] = timestamps

    return TemporalEmbeddingDataset(embedding_file=None, metadata_file=metadata_file)


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
        name_cols = [c for c in df.columns if c.endswith('_image_name')]

        print(f"Reading images from {os.path.basename(parquet_path)} ({len(df)} locations)...")
        for row_idx, row in df.iterrows():
            loc_id = f"{shard_id}_r{row_idx:04d}"
            images = []
            for t_idx, (img_col, name_col) in enumerate(zip(img_cols, name_cols)):
                img = PILImage.open(io.BytesIO(row[img_col]['bytes'])).convert('RGB')
                images.append(img)
                timestamp = parse_date_from_filename(row[name_col])
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
