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
