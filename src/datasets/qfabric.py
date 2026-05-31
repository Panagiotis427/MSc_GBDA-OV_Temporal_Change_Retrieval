"""
QFabric loader, packaged as a `TemporalDataset`.

Wraps the existing `load_parquet_as_embeddings` from `src/data_loader.py`. The
QFabric parquet shards expose 5 fixed timepoints per location row; pairs are
consecutive: ``(t1,t2), (t2,t3), (t3,t4), (t4,t5)`` → 4 pairs per location.

This class is intentionally a thin adapter: existing tests against
`load_parquet_as_embeddings` keep working, while new code can talk to the
`TemporalDataset` protocol.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

from .base import PairKey, PairLabel


class QFabricDataset:
    """`TemporalDataset` implementation backed by QFabric parquet shards.

    Three construction modes:

    1. ``parquet_paths`` + ``encoder`` provided → encode on first call
       (re-uses `load_parquet_as_embeddings` for compatibility).
    2. ``parquet_paths`` + existing ``cache_path`` → loads cached embeddings
       only (no encoder needed).
    3. ``embedding_lookup`` / ``image_lookup`` / ``metadata_df`` passed
       directly → trivial wrapper around pre-built state (useful for tests).
    """

    name = "qfabric"
    temporal_axis_type = "fixed-5"
    TIMEPOINT_KEYS: Tuple[str, ...] = ("t1", "t2", "t3", "t4", "t5")

    def __init__(
        self,
        parquet_paths: Optional[Union[str, List[str]]] = None,
        encoder: Optional[Any] = None,
        cache_path: Optional[str] = None,
        device: Optional[Any] = None,
        *,
        images_only: bool = False,
        embedding_lookup: Optional[Dict[str, np.ndarray]] = None,
        image_lookup: Optional[Dict[str, List[Image.Image]]] = None,
        metadata_df: Optional[pd.DataFrame] = None,
    ) -> None:
        if embedding_lookup is not None and metadata_df is not None:
            self._embedding_lookup = embedding_lookup
            self._image_lookup = image_lookup or {}
            self._metadata_df = metadata_df
        elif images_only:
            # Project pipeline path: read PIL images from the parquet shards
            # *without* any CLIP encoding. ``embeddings.load_or_compute`` then
            # encodes them with the selected project encoder (clip_vitl14 /
            # georsclip / remoteclip), matching the DEN code path exactly.
            if parquet_paths is None:
                raise ValueError("QFabricDataset(images_only=True) needs parquet_paths.")
            self._image_lookup, self._metadata_df = self._read_parquet_images(parquet_paths)
            # zero-width arrays carry only the per-location timepoint count that
            # ``list_pairs`` needs; the real vectors come from the project encoder.
            self._embedding_lookup = {
                loc: np.zeros((len(imgs), 0), dtype=np.float32)
                for loc, imgs in self._image_lookup.items()
            }
        else:
            if parquet_paths is None:
                raise ValueError(
                    "QFabricDataset requires either parquet_paths or "
                    "(embedding_lookup, metadata_df) to be provided."
                )
            self._embedding_lookup, self._metadata_df, self._image_lookup = self._build_from_parquet(
                parquet_paths=parquet_paths,
                encoder=encoder,
                cache_path=cache_path,
                device=device,
            )

        self._enrich_metadata()
        self._pairs_cache: Optional[List[PairKey]] = None

    @staticmethod
    def _read_parquet_images(
        parquet_paths: Union[str, List[str]],
    ) -> Tuple[Dict[str, List[Image.Image]], pd.DataFrame]:
        """Decode the 5 per-row images from each shard into a location->images
        lookup + a metadata frame. No model, no encoding (fast PNG decompress)."""
        import io
        import os

        from ..data_loader import parse_date_from_filename

        if isinstance(parquet_paths, str):
            parquet_paths = [parquet_paths]
        image_lookup: Dict[str, List[Image.Image]] = {}
        rows: List[Dict[str, Any]] = []
        for pp in parquet_paths:
            shard = os.path.splitext(os.path.basename(pp))[0]
            df = pd.read_parquet(pp)
            img_cols = [c for c in df.columns
                        if c.endswith("_image") and not c.endswith("_name")]
            name_cols = [c for c in df.columns if c.endswith("_image_name")]
            for ri, row in df.iterrows():
                loc = f"{shard}_r{int(ri):04d}"
                imgs: List[Image.Image] = []
                for ti, (ic, nc) in enumerate(zip(img_cols, name_cols)):
                    imgs.append(Image.open(io.BytesIO(row[ic]["bytes"])).convert("RGB"))
                    rows.append({"location": loc,
                                 "timestamp": parse_date_from_filename(row[nc]),
                                 "timepoint_idx": ti})
                image_lookup[loc] = imgs
        return image_lookup, pd.DataFrame(rows)

    @staticmethod
    def _build_from_parquet(
        parquet_paths: Union[str, List[str]],
        encoder: Optional[Any],
        cache_path: Optional[str],
        device: Optional[Any],
    ) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict[str, List[Image.Image]]]:
        from ..data_loader import load_parquet_as_embeddings

        if encoder is not None:
            clip_model = getattr(encoder, "_clip_model", None)
            processor = getattr(encoder, "_processor", None)
            if clip_model is None or processor is None:
                raise ValueError(
                    "Encoder does not expose `_clip_model` and `_processor` "
                    "attributes required by the legacy QFabric loader. "
                    "Either pass a CLIPViTL14Encoder or supply a cache_path."
                )
        else:
            clip_model = processor = None

        return load_parquet_as_embeddings(
            parquet_paths=parquet_paths,
            clip_model=clip_model,
            processor=processor,
            device=device,
            cache_path=cache_path,
        )

    def _enrich_metadata(self) -> None:
        """Ensure metadata has the columns the Protocol requires."""
        df = self._metadata_df.copy()
        if "t_key" not in df.columns:
            if "timepoint_idx" in df.columns:
                df["t_key"] = df["timepoint_idx"].apply(
                    lambda i: self.TIMEPOINT_KEYS[int(i)] if 0 <= int(i) < len(self.TIMEPOINT_KEYS) else f"t{int(i)+1}"
                )
            else:
                df["t_key"] = "t1"
        if "pair_id" not in df.columns:
            df["pair_id"] = df["location"].astype(str) + "::" + df["t_key"]
        if "dataset_name" not in df.columns:
            df["dataset_name"] = self.name
        self._metadata_df = df

    def list_locations(self) -> List[str]:
        return sorted(self._embedding_lookup.keys())

    def list_pairs(self) -> List[PairKey]:
        if self._pairs_cache is not None:
            return self._pairs_cache
        pairs: List[PairKey] = []
        for loc in self.list_locations():
            n_timepoints = len(self._embedding_lookup[loc])
            for i in range(n_timepoints - 1):
                t1 = self.TIMEPOINT_KEYS[i] if i < len(self.TIMEPOINT_KEYS) else f"t{i+1}"
                t2 = self.TIMEPOINT_KEYS[i + 1] if (i + 1) < len(self.TIMEPOINT_KEYS) else f"t{i+2}"
                pairs.append(PairKey(location_id=loc, t1_key=t1, t2_key=t2))
        self._pairs_cache = pairs
        return pairs

    def load_image(self, location_id: str, t_key: str) -> Image.Image:
        images = self._image_lookup.get(location_id)
        if not images:
            raise KeyError(
                f"No images cached for location '{location_id}'. "
                f"Construct QFabricDataset with parquet_paths to populate the image lookup."
            )
        idx = self._t_key_to_index(t_key)
        if idx >= len(images):
            raise IndexError(
                f"t_key '{t_key}' resolves to index {idx} but only {len(images)} images for {location_id}"
            )
        return images[idx]

    def load_pair_images(self, pair: PairKey) -> Tuple[Image.Image, Image.Image]:
        return self.load_image(pair.location_id, pair.t1_key), self.load_image(pair.location_id, pair.t2_key)

    def load_metadata(self) -> pd.DataFrame:
        return self._metadata_df

    def get_pair_label(self, pair: PairKey) -> Optional[PairLabel]:
        return None

    @property
    def embedding_lookup(self) -> Dict[str, np.ndarray]:
        """Direct access to the underlying ``{location_id: [n_t, embed_dim]}`` dict.

        Required by `pair_iter_from_dataset` and the existing FAISS index builder.
        """
        return self._embedding_lookup

    def _t_key_to_index(self, t_key: str) -> int:
        if t_key in self.TIMEPOINT_KEYS:
            return self.TIMEPOINT_KEYS.index(t_key)
        if t_key.startswith("t") and t_key[1:].isdigit():
            return int(t_key[1:]) - 1
        raise ValueError(f"Unrecognised t_key '{t_key}' for QFabric")


def from_cache(cache_path: str, parquet_paths: Optional[Union[str, List[str]]] = None) -> QFabricDataset:
    """Convenience: build a QFabricDataset directly from a cached `.npz`.

    If ``parquet_paths`` is given, also reads them for the image lookup (needed
    only for heatmap rendering and the Gradio gallery).
    """
    if parquet_paths is None:
        data = np.load(cache_path)
        embedding_lookup = {k: data[k] for k in data.files}
        metadata_rows = []
        for loc, emb in embedding_lookup.items():
            for t_idx in range(len(emb)):
                metadata_rows.append({
                    "location": loc,
                    "timestamp": pd.Timestamp("2015-01-01") + pd.Timedelta(days=t_idx),
                    "timepoint_idx": t_idx,
                })
        metadata_df = pd.DataFrame(metadata_rows)
        return QFabricDataset(embedding_lookup=embedding_lookup, metadata_df=metadata_df)

    return QFabricDataset(parquet_paths=parquet_paths, cache_path=cache_path)
