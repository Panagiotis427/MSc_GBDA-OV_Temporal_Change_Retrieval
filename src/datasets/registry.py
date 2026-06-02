"""
Dataset factory + option-adapter registry.

Two registries keyed by dataset short-name:

- ``_FACTORIES[name]``: callable that builds a ``TemporalDataset`` from
  loader-specific kwargs.
- ``_OPTS[name]``: callable that maps the project's *generic* CLI / pipeline
  options (``root``, ``pairing``, ``split``, ``**extra``) onto the kwargs that
  loader expects.

Pipeline callers (``embeddings`` / ``benchmark`` / ``train`` / ``app`` /
``run_pipeline``) use :func:`build_dataset` and never branch on dataset name.
Adding a dataset = register a factory + an opts adapter in a new module; no
shared file needs editing.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .base import TemporalDataset


# Lazy: each entry is a zero-import callable that imports its loader on first
# use, avoiding heavy deps (open_clip / rasterio / parquet) for unused loaders.
_FACTORIES: Dict[str, Callable[..., TemporalDataset]] = {}
_OPTS: Dict[str, Callable[..., Dict[str, Any]]] = {}


def register_dataset(
    name: str,
    factory: Callable[..., TemporalDataset],
    opts: Optional[Callable[..., Dict[str, Any]]] = None,
) -> None:
    """Register *name* -> (factory, opts adapter)."""
    _FACTORIES[name] = factory
    _OPTS[name] = opts or _default_opts


def list_datasets() -> list[str]:
    return sorted(_FACTORIES)


def get_dataset(name: str, **kwargs: Any) -> TemporalDataset:
    """Instantiate by name with loader-specific kwargs (back-compat path)."""
    if name not in _FACTORIES:
        raise ValueError(
            f"Unknown dataset: '{name}'. Registered: {list_datasets()}"
        )
    return _FACTORIES[name](**kwargs)


def build_dataset(
    name: str,
    *,
    root: Optional[str] = None,
    pairing: Optional[str] = None,
    split: Optional[str] = None,
    **extra: Any,
) -> TemporalDataset:
    """Generic entry point used by the whole pipeline.

    Generic options are translated to loader-specific kwargs by the dataset's
    opts adapter. ``split="all"`` is normalised to ``None``.
    """
    if name not in _FACTORIES:
        raise ValueError(
            f"Unknown dataset: '{name}'. Registered: {list_datasets()}"
        )
    if split == "all":
        split = None
    generic = {"root": root, "pairing": pairing, "split": split, **extra}
    kwargs = _OPTS[name](**generic)
    return _FACTORIES[name](**kwargs)


def _default_opts(*, root=None, pairing=None, split=None, **extra) -> Dict[str, Any]:
    """Pass-through: include only the generic options the loader is likely to
    accept; concrete loaders can register their own adapter for full control."""
    out: Dict[str, Any] = {}
    if root is not None:
        out["root"] = root
    out.update(extra)
    return out


# ----------------------------------------------------------------------
# Built-in dataset registrations
# ----------------------------------------------------------------------

def _qfabric_factory(**kwargs: Any) -> TemporalDataset:
    from .qfabric import QFabricDataset
    return QFabricDataset(**kwargs)


def _qfabric_opts(*, root=None, pairing=None, split=None, **extra) -> Dict[str, Any]:
    """QFabric needs ``parquet_paths`` or a precomputed embedding store. By
    default, treat ``root`` as a directory holding ``*.parquet`` shards; any
    explicit kwarg in ``extra`` (``parquet_paths``, ``cache_path``,
    ``embedding_lookup`` ...) wins.

    Generic pipeline options the QFabric loader doesn't accept are dropped:
    ``color_mode`` (parquet is RGB-only), ``pairing``/``split`` (fixed-5 axis,
    no splits). Defaults to ``images_only`` so the project encoder does the
    encoding via ``load_or_compute`` (same path as DEN), not the legacy
    CLIP-in-loader path.
    """
    extra.pop("color_mode", None)  # QFabric parquet is RGB-only
    out: Dict[str, Any] = dict(extra)
    if root and "parquet_paths" not in out and not any(
        k in out for k in ("embedding_lookup", "cache_path")
    ):
        import glob
        import os
        shards = sorted(glob.glob(os.path.join(root, "*.parquet")))
        if shards:
            out["parquet_paths"] = shards
            out.setdefault("images_only", True)
    return out


def _qfabric_teo_factory(**kwargs: Any) -> TemporalDataset:
    from .qfabric_teo import TEOChatlasQFabricDataset
    return TEOChatlasQFabricDataset(**kwargs)


def _qfabric_teo_opts(*, root=None, pairing=None, split=None, **extra) -> Dict[str, Any]:
    """TEOChatlas-QFabric: ``root`` is a dir of extracted QFabric ``.tif`` crops.
    Drops generic kwargs the loader ignores (color_mode/pairing/split — RGB,
    fixed before/after axis, no splits). ``max_per_class`` / ``labels_path`` /
    ``seed`` pass through via extra."""
    extra.pop("color_mode", None)
    out: Dict[str, Any] = dict(extra)
    if root is not None:
        out["root"] = root
    if split is not None:           # train|test partition (None/"all" => whole corpus)
        out["split"] = split
    return out


def _qfabric_status_factory(**kwargs: Any) -> TemporalDataset:
    from .qfabric_status import StatusQFabricDataset
    return StatusQFabricDataset(**kwargs)


def _qfabric_status_opts(*, root=None, pairing=None, split=None, **extra) -> Dict[str, Any]:
    """TEOChatlas-QFabric status-transition (RQA5): same option mapping as the
    RQA2 loader — ``root`` is the extracted crops dir; ``max_per_class`` /
    ``labels_path`` / ``seed`` pass through; ``split`` selects train|test."""
    extra.pop("color_mode", None)
    out: Dict[str, Any] = dict(extra)
    if root is not None:
        out["root"] = root
    if split is not None:
        out["split"] = split
    return out


def _dynamic_earthnet_factory(**kwargs: Any) -> TemporalDataset:
    """Auto-detect on-disk layout: the preprocessed DynNet gdown subset
    (``labels/*.npy`` + ``split.json``) vs the raster ``planet/<aoi>/*.tif``
    layout (and the synthetic test fixture)."""
    from .dynamic_earthnet_pp import resolve_pp_root

    root = kwargs.get("root")
    if root is not None and resolve_pp_root(root) is not None:
        from .dynamic_earthnet_pp import DENNpyDataset
        return DENNpyDataset(**kwargs)
    from .dynamic_earthnet import DENDataset
    kwargs.pop("split", None)      # raster loader has no split arg
    kwargs.pop("color_mode", None) # raster loader has no color_mode (single RGB)
    return DENDataset(**kwargs)


def _dynamic_earthnet_opts(*, root=None, pairing=None, split=None, **extra) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(extra)
    if root is not None:
        out["root"] = root
    out["pairing_strategy"] = pairing or "bimonthly"
    if split is not None:
        out["split"] = split
    return out


register_dataset("qfabric", _qfabric_factory, _qfabric_opts)
register_dataset("qfabric_teo", _qfabric_teo_factory, _qfabric_teo_opts)
register_dataset("qfabric_status", _qfabric_status_factory, _qfabric_status_opts)
register_dataset("dynamic_earthnet", _dynamic_earthnet_factory, _dynamic_earthnet_opts)
