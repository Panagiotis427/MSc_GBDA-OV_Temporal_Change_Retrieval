"""
Multi-timepoint temporal sequence analysis for change detection.

This module extends the T1/T2 pairing from Weeks 1-5 to handle sliding windows,
multiple timepoints (T-3, T-2, T-1 → current), and seasonal pattern recognition.
Used by Week 6 error analysis to distinguish snow melting vs permanent construction.
"""
from typing import Dict, List, Generator, Tuple, Optional
import numpy as np
import pandas as pd


def create_triplet_sequences(
    metadata_df: pd.DataFrame,
    embedding_lookup: Dict[str, np.ndarray],
    window_size: int = 3,
    stride: int = 1
) -> Generator[Tuple[pd.Series, Tuple[np.ndarray, ...]], None, None]:
    """
    Create sliding window sequences of embeddings for seasonal pattern detection.

    For each location, generates overlapping windows like:
    - Window 0: [T-2, T-1, current]
    - Window 1: [T-1, current, future]
    ...

    Args:
        metadata_df (pd.DataFrame): Sorted DataFrame with ['location', 'timestamp'] columns.
        embedding_lookup (dict): Maps location_id -> numpy array of embeddings.
        window_size (int): Number of timepoints per sequence (default 3 for triplets).
        stride (int): Step between consecutive windows within a location.

    Yields:
        tuple: (window_timestamps, (emb_1, emb_2, ..., emb_n))
            - window_timestamps: Series with timestamps of current window
            - emb_tuple: Tuple of embedding arrays for each timepoint in window
    """
    grouped = metadata_df.groupby('location')

    for loc, group in grouped:
        n_obs = len(group)
        if n_obs < window_size:
            continue  # Not enough observations for a full window

        emb_array = embedding_lookup[loc]

        # Create overlapping windows
        for start_idx in range(0, n_obs - window_size + 1, stride):
            end_idx = start_idx + window_size
            timestamps = group.iloc[start_idx:end_idx]['timestamp']
            embeddings = tuple(emb_array[i] for i in range(start_idx, end_idx))
            yield timestamps, embeddings


def create_history_context(
    metadata_df: pd.DataFrame,
    embedding_lookup: Dict[str, np.ndarray],
    history_days: int = 365
) -> Generator[Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]], None, None]:
    """
    Get historical embeddings within a time window for seasonal comparison.

    For each current T2 observation, finds the most recent embedding from ~1 year ago,
    then compares seasonal patterns (e.g., same season last year vs this year).

    Args:
        metadata_df (pd.DataFrame): Sorted DataFrame with ['location', 'timestamp'] columns.
        embedding_lookup (dict): Maps location_id -> numpy array of embeddings.
        history_days (int): Lookback period in days for seasonal comparison. Default 365.

    Yields:
        tuple: (current_emb, historical_emb, [historical_timestamp])
            - current_emb: Current observation embedding
            - historical_emb: Historical embedding from ~1 year ago (or nearest)
            - historical_timestamps: List of all timestamps within history window
    """
    grouped = metadata_df.groupby('location')

    for loc, group in grouped:
        emb_array = embedding_lookup[loc]
        n_obs = len(group)

        # Ensure we have enough history
        if n_obs < 3:  # Need at least current + 2 historical points
            continue

        # For each T2 observation, find matching seasonal point from ~1 year ago
        for i in range(1, min(n_obs - 1, int(n_obs / 2))):  # Skip first (T0), do pairs
            t2_idx = n_obs - i  # Current T2
            historical_idx = max(0, t2_idx - int(history_days / 30))  # ~1 year back

            if historical_idx >= t2_idx:
                continue  # Not enough history

            current_emb = emb_array[t2_idx]
            hist_emb = emb_array[historical_idx]
            hist_ts = group.iloc[:t2_idx]['timestamp'].tolist()

            yield current_emb, hist_emb, hist_ts


def compute_temporal_variance(
    embeddings: np.ndarray,
    timestamps: pd.Series
) -> Tuple[float, float]:
    """
    Compute temporal variance metrics for a single location's embedding sequence.

    Used to detect stability vs volatility of features over time:
    - Low variance = seasonal (predictable changes like snow/leaves)
    - High variance = permanent or random

    Args:
        embeddings (np.ndarray): 2D array of shape (N, dim) for one location.
        timestamps (pd.Series): Timestamps corresponding to each embedding.

    Returns:
        tuple: (temporal_variance, stability_score)
            - temporal_variance: Variance of cosine similarity between consecutive pairs
            - stability_score: 0-1 metric where low = volatile, high = stable/seasonal
    """
    n_obs = len(embeddings)
    if n_obs < 2:
        return 0.0, 0.0

    # Compute cosine similarities between consecutive embeddings
    emb_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm[emb_norm == 0] = 1  # Avoid division by zero
    cos_sim = np.dot(embeddings, embeddings.T) / (np.outer(emb_norm, emb_norm))

    # Variance of consecutive similarities - high variance = unstable changes
    consecutive_sims = [cos_sim[i, i+1] for i in range(n_obs - 1)]
    temporal_var = np.var(consecutive_sims)

    # Stability score: inverse relationship with variance (capped at 1.0)
    stability_score = 1.0 / (1.0 + temporal_var) * 50  # Scale to reasonable range
    if stability_score > 1.0:
        stability_score = 1.0

    return float(temporal_var), float(stability_score)


def compute_seasonal_consistency(
    window_sequences: Generator[Tuple[pd.Series, Tuple[np.ndarray, ...]], None, None],
    reference_emb: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute seasonal consistency metrics across multiple timepoint windows.

    For detecting recurring patterns (e.g., snow every winter vs one-time construction).
    Uses cosine similarity between all pairs in a window to measure how consistent
    the change is relative to baseline.

    Args:
        window_sequences: Generator from create_triplet_sequences()
        reference_emb: Optional single embedding to compare windows against (e.g., pre-change state)

    Returns:
        dict with metrics:
            - 'mean_pairwise_sim': Average cosine similarity across all timepoint pairs in window
            - 'window_consistency': How consistent the change is relative to baseline
            - 'temporal_coherence': Correlation between embeddings over the sequence
    """
    import math

    metrics = {
        'mean_pairwise_sim': [],
        'window_consistency': [],
        'temporal_coherence': []
    }

    for timestamps, emb_tuple in window_sequences:
        if len(emb_tuple) < 2:
            continue

        # Compute mean pairwise cosine similarity within window
        pair_sims = []
        for i in range(len(emb_tuple)):
            for j in range(i+1, len(emb_tuple)):
                sim = np.dot(emb_tuple[i], emb_tuple[j]) / (
                    (np.linalg.norm(emb_tuple[i]) + 1e-8) *
                    (np.linalg.norm(emb_tuple[j]) + 1e-8)
                )
                pair_sims.append(sim)

        if not pair_sims:
            continue

        metrics['mean_pairwise_sim'].append(np.mean(pair_sims))

        # Window consistency: how much the window deviates from reference (or median of all windows)
        if reference_emb is not None:
            ref_norm = np.linalg.norm(reference_emb) + 1e-8
            window_mean = np.mean([np.dot(e, reference_emb) / ref_norm
                                  for e in emb_tuple])
            metrics['window_consistency'].append(window_mean)
        else:
            # Use median across all windows as baseline
            mean_sim = np.mean(pair_sims)
            metrics['temporal_coherence'].append(mean_sim - 0.5)  # Center around 0.5 (random baseline)

    # Aggregate metrics
    if any(v for v in metrics.values()):
        return {
            'mean_pairwise_sim': float(np.mean(metrics['mean_pairwise_sim'])),
            'window_consistency': float(np.mean(metrics['window_consistency']) if metrics['window_consistency'] else None),
            'temporal_coherence': float(np.mean(metrics['temporal_coherence']))
        }
    return {}


def create_baseline_dataset(
    metadata_df: pd.DataFrame,
    embedding_lookup: Dict[str, np.ndarray],
    baseline_window_days: int = 180  # ~6 months for seasonal stability measurement
) -> Dict[str, Dict]:
    """
    Create a reference dataset of stable (seasonal-only) changes.

    Used as ground truth to calibrate seasonal vs permanent classifiers:
    - Snow melting: high temporal consistency over multiple years
    - Leaf fall: consistent pattern every spring/autumn
    - Permanent construction: different consistency profile

    Args:
        metadata_df (pd.DataFrame): Full dataset with all observations.
        embedding_lookup (dict): Location -> embeddings mapping.
        baseline_window_days: Window size for measuring stability. Default 180 days.

    Returns:
        dict: Per-location statistics showing seasonal vs permanent change characteristics
            - 'location_id': str
            - 'temporal_var': float from compute_temporal_variance()
            - 'window_count': number of sliding windows computed
    """
    results = {}
    grouped = metadata_df.groupby('location')

    for loc, group in grouped:
        emb_array = embedding_lookup[loc]
        n_obs = len(group)

        # Compute baseline variance using long window (seasonal patterns should be stable)
        var, stability = compute_temporal_variance(emb_array, group['timestamp'])

        results[loc] = {
            'temporal_var': var,
            'stability_score': stability,
            'observation_count': n_obs
        }

    return results