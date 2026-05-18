"""
Geographic filtering for pair retrieval.

Loads ``aoi_metadata.json`` (WGS84 bboxes per AOI, computed by
``scripts/run_pipeline.py``) and exposes helpers to restrict a list of
:class:`~src.datasets.base.PairKey` objects to a geographic region or
bounding box.  Used by the Gradio app; the filter is optional and can be
disabled at any time by selecting region "All".
"""
from __future__ import annotations

import json
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.datasets.base import PairKey

REGION_ALL = "All"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(max(0.0, min(1.0, a))))


class GeoFilter:
    """Filter :class:`PairKey` lists by geographic region or bounding box.

    Parameters
    ----------
    metadata_path:
        Path to ``aoi_metadata.json``.  Each key is a ``location_id``; each
        value must contain at least ``lat_c``, ``lon_c``, and ``region``.
    """

    def __init__(self, metadata_path: str | Path) -> None:
        with open(metadata_path) as fh:
            self._meta: Dict[str, dict] = json.load(fh)

    # ------------------------------------------------------------------
    @property
    def regions(self) -> List[str]:
        """Sorted list of unique region strings, prepended with ``"All"``."""
        unique = sorted({v["region"] for v in self._meta.values() if "region" in v})
        return [REGION_ALL] + unique

    def centroid(self, location_id: str) -> Optional[Tuple[float, float]]:
        """Return ``(lat_c, lon_c)`` for a location, or ``None`` if unknown."""
        m = self._meta.get(location_id)
        if m is None:
            return None
        lat, lon = m.get("lat_c"), m.get("lon_c")
        if lat is None or lon is None:
            return None
        return float(lat), float(lon)

    # ------------------------------------------------------------------
    def filter_by_region(self, pairs: List[PairKey], region: str) -> List[PairKey]:
        """Return pairs whose AOI belongs to *region*.

        Pairs with unknown ``location_id`` (not in the metadata) are **excluded**
        when a specific region is selected, since we cannot confirm membership.
        Passing ``"All"`` or an empty string returns *pairs* unchanged.
        """
        if region in (REGION_ALL, "", None):
            return pairs
        allowed: Set[str] = {
            loc for loc, m in self._meta.items() if m.get("region") == region
        }
        return [p for p in pairs if p.location_id in allowed]

    def filter_by_bbox(
        self,
        pairs: List[PairKey],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> List[PairKey]:
        """Return pairs whose AOI centroid falls within the given bbox.

        Pairs with unknown location are **kept** (benefit of the doubt).
        """
        def _in(loc_id: str) -> bool:
            c = self.centroid(loc_id)
            if c is None:
                return True
            c_lat, c_lon = c
            return lat_min <= c_lat <= lat_max and lon_min <= c_lon <= lon_max

        return [p for p in pairs if _in(p.location_id)]

    def nearest(
        self,
        location_id: str,
        candidates: List[str],
        top_n: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return the *top_n* nearest location IDs (by haversine) to *location_id*.

        Returns a list of ``(location_id, distance_km)`` tuples, sorted ascending.
        Locations missing from metadata are skipped.
        """
        anchor = self.centroid(location_id)
        if anchor is None:
            return []
        a_lat, a_lon = anchor
        dists: List[Tuple[str, float]] = []
        for loc in candidates:
            if loc == location_id:
                continue
            c = self.centroid(loc)
            if c is None:
                continue
            dists.append((loc, _haversine_km(a_lat, a_lon, c[0], c[1])))
        dists.sort(key=lambda x: x[1])
        return dists[:top_n]
