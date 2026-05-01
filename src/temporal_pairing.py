"""
Temporal pairing logic for multitemporal satellite imagery.

This module provides utilities to group image embeddings by geographic location,
sort them temporally, and yield (T1, T2) pairs for the same spatial tile.
"""
from typing import Dict, List, Generator, Tuple
import pandas as pd
import numpy as np


def group_by_location(
    locations: np.ndarray,
    embeddings: np.ndarray
) -> Generator[Tuple[str, int, int], None, None]:
    """
    Group embeddings by location and yield consecutive time pair indices.

    Args:
        locations (np.ndarray): 1D array of length N containing location identifiers.
        embeddings (np.ndarray): 2D array of shape (N, embedding_dim).

    Yields:
        tuple: (location_id, t1_index, t2_index) where indices are positions in embeddings.

    Example:
        >>> locations = np.array(['A', 'B', 'A', 'B', 'A'])
        >>> list(group_by_location(locations, np.random.randn(5, 10)))
        [('A', 0, 2), ('B', 1, 3), ('A', 4, ?)]  # Last location has only one observation
    """
    unique_locs = np.unique(locations)

    for loc in sorted(unique_locs):
        # Get all indices for this location
        local_indices = np.where(locations == loc)[0]

        if len(local_indices) < 2:
            # Skip locations with insufficient observations
            continue

        yield loc, int(local_indices[0]), int(local_indices[-1])


def pair_temporally(
    metadata_df: pd.DataFrame,
    embedding_lookup: Dict[str, np.ndarray]
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yield (embedding_T1, embedding_T2) pairs sorted by location and timestamp.

    Args:
        metadata_df (pd.DataFrame): DataFrame with columns ['location', 'timestamp']. Must be sorted
            by (location, timestamp).
        embedding_lookup (dict): Maps location_id -> numpy array of embeddings.

    Yields:
        tuple: Two 1D arrays (emb_t1, emb_t2) representing consecutive time steps for the same tile.

    Raises:
        ValueError: If metadata is not sorted or if any location has odd number of observations.
    """
    # Validate sorting
    prev_loc = None
    prev_rank = 0

    for idx, (loc, ts) in enumerate(zip(metadata_df['location'], metadata_df['timestamp'])):
        if prev_loc is not None and loc < prev_loc:
            raise ValueError("Metadata must be sorted by location, timestamp. Found out-of-order entry.")

        if loc == prev_loc:
            # Same location - check temporal ordering
            current_rank = (ts - metadata_df.loc[idx-1, 'timestamp']).days
            if current_rank < 0:
                raise ValueError(f"Timestamps not monotonically increasing at index {idx}")

        prev_loc, prev_rank = loc, current_rank if prev_loc else 0

    # Group by location and create pairs
    grouped = metadata_df.groupby('location')

    for loc, group in grouped:
        n_obs = len(group)
        if n_obs % 2 != 0:
            raise ValueError(f"Location {loc} has odd number of observations ({n_obs}). Cannot form complete pairs.")

        # Create consecutive pairs
        for i in range(0, n_obs, 2):
            t1_emb = embedding_lookup[loc][i]
            t2_emb = embedding_lookup[loc][i+1]
            yield t1_emb, t2_emb


def create_pair_indices(
    metadata_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create integer indices that group temporal pairs together in a dataset.

    This is useful when you want to use PyTorch's DataLoader with batch_size > 1,
    ensuring each batch contains complete (T1, T2) pairs for the same location.

    Args:
        metadata_df (pd.DataFrame): Sorted DataFrame with ['location', 'timestamp'] columns.

    Returns:
        tuple: (group_indices, group_ids)
            - group_indices: 1D array where each position corresponds to a row in metadata_df,
              and identical values indicate observations belonging to the same (T1, T2) pair.
            - group_ids: 1D array of unique identifiers for each location-group. Use as batch keys.
    """
    # Assign group ID to each observation
    grouped = metadata_df.groupby('location')

    group_id_list = []
    current_group = None
    pair_counter = {}

    for loc, group in grouped:
        n_obs = len(group)
        if n_obs % 2 != 0:
            raise ValueError(f"Location {loc} has odd observations")

        # Assign same group ID to both T1 and T2
        pair_id = f"{loc}_pair"
        for i, (_, obs) in enumerate(group.iterrows()):
            current_group = pair_id if current_group is None else current_group
            group_id_list.append(pair_id)

    # Convert to numpy arrays
    group_ids = np.array(sorted(set(group_id_list)))
    id_to_idx = {gid: i for i, gid in enumerate(group_ids)}

    return np.array([id_to_idx[g] for g in group_id_list]), np.array(group_id_list)


def validate_temporal_dataset(
    metadata_df: pd.DataFrame,
    embedding_lookup: Dict[str, np.ndarray]
) -> Tuple[int, int]:
    """
    Validate that the dataset can form complete temporal pairs.

    Args:
        metadata_df (pd.DataFrame): DataFrame with ['location', 'timestamp'] columns.
        embedding_lookup (dict): Maps location_id -> numpy array of embeddings.

    Returns:
        tuple: (total_pairs, total_observations)
    """
    total_obs = 0
    total_pairs = 0

    for loc in metadata_df['location'].unique():
        n_obs = len(metadata_df[metadata_df['location'] == loc])
        if n_obs % 2 != 0:
            raise ValueError(f"Location {loc} has odd number of observations")

        total_obs += n_obs
        total_pairs += n_obs // 2

    # Verify embedding arrays exist and have correct lengths
    for loc in metadata_df['location'].unique():
        if loc not in embedding_lookup:
            raise KeyError(f"No embeddings found for location {loc}")

        emb_length = len(embedding_lookup[loc])
        n_obs = len(metadata_df[metadata_df['location'] == loc])
        # expected = n_obs  # Walrus operator for Python 3.8+ compatibility
        if emb_length != n_obs:
            raise ValueError(
                f"Embeddings for {loc} have length {emb_length}, "
                f"but metadata has {n_obs} observations"
            )

    return total_pairs, total_obs
