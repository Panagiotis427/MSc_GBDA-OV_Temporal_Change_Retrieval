"""
Dataset-agnostic loader subpackage.

Public API:

    from src.datasets import get_dataset, TemporalDataset, PairKey, PairLabel

Concrete dataset classes:

    from src.datasets.qfabric import QFabricDataset
    from src.datasets.dynamic_earthnet import DENDataset   # session 2
"""
from .base import (
    PairKey,
    PairLabel,
    TemporalDataset,
    metadata_from_dataset,
    pair_iter_from_dataset,
)
from .registry import get_dataset, register_dataset

__all__ = [
    "PairKey",
    "PairLabel",
    "TemporalDataset",
    "get_dataset",
    "register_dataset",
    "metadata_from_dataset",
    "pair_iter_from_dataset",
]
