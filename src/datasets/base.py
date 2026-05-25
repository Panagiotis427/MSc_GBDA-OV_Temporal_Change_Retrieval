"""
Dataset-agnostic core for temporal change retrieval.

Defines a structural Protocol (`TemporalDataset`) that every dataset loader
must satisfy, plus the lightweight value types (`PairKey`, `PairLabel`) that
move through the rest of the pipeline.

Using `typing.Protocol` (not ABC) so existing dict-based loaders can be
adapted via thin wrappers without forced inheritance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import numpy as np
import pandas as pd
from PIL import Image


class PairKey(NamedTuple):
    """Identifies a single bi-temporal pair within a dataset.

    Attributes:
        location_id: Identifier of the spatial tile / AOI.
        t1_key: Encoded time-step identifier for T1 (e.g. ``"2018-01-01"``,
            ``"t1"``, or ``"timepoint_0"``). The encoding is dataset-specific.
        t2_key: Same as ``t1_key`` for the second observation.
    """

    location_id: str
    t1_key: str
    t2_key: str


@dataclass
class PairLabel:
    """Ground-truth annotation for a bi-temporal pair.

    Returned by `TemporalDataset.get_pair_label`. May be `None` for unlabeled
    pairs; implementations should return `None` rather than a default-filled
    instance to make label availability explicit.

    Attributes:
        change_type: Coarse string label such as ``"stable"`` or
            ``"forest->impervious_surface"``.
        stable: True iff the pair shows no significant change (definition is
            dataset-specific; for DEN we use ``total_change < stable_threshold``).
        dominant_t1_class: Most frequent class in T1's label tile (if labels are
            pixel-wise) or None.
        dominant_t2_class: Same for T2.
        class_change_mask_fraction: Per-class ``{"gained_fraction": float,
            "lost_fraction": float}`` summary. Empty for snapshot datasets.
    """

    change_type: str
    stable: bool
    dominant_t1_class: Optional[str] = None
    dominant_t2_class: Optional[str] = None
    class_change_mask_fraction: Dict[str, Dict[str, float]] = field(default_factory=dict)


@runtime_checkable
class TemporalDataset(Protocol):
    """Protocol for any dataset usable as a retrieval corpus.

    Downstream code (`temporal_pairing.pair_temporally_from_dataset`, the
    Gradio app, the training loop) only depends on this interface.

    Required attributes:
        name: Short identifier, e.g. ``"qfabric"``, ``"dynamic_earthnet"``.
        temporal_axis_type: One of ``"fixed-5"`` | ``"daily"`` | ``"snapshot"``.
            Hint for the pairing strategy; not enforced.
    """

    name: str
    temporal_axis_type: str

    # Discovery
    def list_locations(self) -> List[str]: ...
    def list_pairs(self) -> List[PairKey]: ...

    # Data access (lazy)
    def load_image(self, location_id: str, t_key: str) -> Image.Image: ...
    def load_pair_images(self, pair: PairKey) -> Tuple[Image.Image, Image.Image]: ...

    # Metadata
    def load_metadata(self) -> pd.DataFrame:
        """Return a DataFrame with the columns the rest of the pipeline relies on.

        Required columns: ``location``, ``timestamp``, ``t_key``, ``pair_id``,
        ``dataset_name``.
        """
        ...

    # Labels (optional per pair)
    def get_pair_label(self, pair: PairKey) -> Optional[PairLabel]: ...


def metadata_from_dataset(dataset: TemporalDataset) -> pd.DataFrame:
    """Convenience helper: realise a dataset's metadata frame, validating it
    has the columns the rest of the pipeline expects.
    """
    df = dataset.load_metadata()
    required = {"location", "timestamp", "t_key", "pair_id", "dataset_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset '{dataset.name}' metadata is missing required columns: {sorted(missing)}"
        )
    return df


def pair_iter_from_dataset(
    dataset: TemporalDataset,
    embedding_lookup: Dict[str, np.ndarray],
) -> Generator[Tuple[PairKey, np.ndarray, np.ndarray], None, None]:
    """Yield ``(pair_key, emb_t1, emb_t2)`` for every pair in the dataset.

    ``embedding_lookup`` is keyed by ``location_id`` with arrays shaped
    ``[n_timepoints, embed_dim]``. Per-location ordering must match the order
    in which time-steps were originally extracted; the loader's `load_metadata`
    DataFrame is authoritative for the index of each ``t_key`` within that
    array.
    """
    metadata = metadata_from_dataset(dataset)
    # location_id -> {t_key: row index within that location}
    rank_within_location: Dict[str, Dict[str, int]] = {}
    for loc, grp in metadata.sort_values(["location", "timestamp"]).groupby("location"):
        rank_within_location[loc] = {t_key: i for i, t_key in enumerate(grp["t_key"].tolist())}

    for pair in dataset.list_pairs():
        emb_array = embedding_lookup.get(pair.location_id)
        if emb_array is None:
            continue
        ranks = rank_within_location.get(pair.location_id, {})
        i1 = ranks.get(pair.t1_key)
        i2 = ranks.get(pair.t2_key)
        if i1 is None or i2 is None or i1 >= len(emb_array) or i2 >= len(emb_array):
            continue
        yield pair, emb_array[i1], emb_array[i2]
