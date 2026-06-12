"""
LEVIR-MCI loader --- LEVIR-CC's bi-temporal pairs and human captions, plus
pixel-level change-detection masks for building and road change.

LEVIR-MCI (Liu et al., *Change-Agent*, IEEE TGRS 2024) is a strict superset of
LEVIR-CC: the same 10,077 image pairs and the same ``LevirCCcaptions.json``, with
an added ``label/`` mask per pair. Layout after extraction (HuggingFace
``lcybuaa/LEVIR-MCI``)::

    <root>/                                  # .../LEVIR-MCI-dataset
      LevirCCcaptions.json
      images/{train,val,test}/A/<name>.png       # pre-phase  (T1)
      images/{train,val,test}/B/<name>.png       # post-phase (T2)
      images/{train,val,test}/label/<name>.png   # change mask (grayscale)
      images/{train,val,test}/label_rgb/<name>.png

Mask encoding (per the dataset's ``readme.txt``), read as a single channel:
``0`` = background / no change, ``128`` = road change, ``255`` = building change.

Retrieval behaviour is identical to ``levir_cc`` (same captions → same tags →
same queries), so the retrieval numbers do not change. The reason to use this
loader instead is :meth:`load_change_mask`, which supplies the ground-truth
localization labels consumed by ``scripts/eval_localization.py`` to turn the
heatmap deliverable from qualitative into quantitative.

Self-registers nothing; the dataset is wired through ``src/datasets/registry.py``
(``levir_mci``), and its retrieval queries reuse the ``levir_cc`` query set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

from .base import PairKey
from .levir_cc import LevirCCDataset

# Mask grayscale value -> change class (readme.txt). Background (0) is implicit.
MASK_VALUE = {"road": 128, "building": 255}

# Caption-derived query text -> mask class. Only building and road carry masks in
# LEVIR-MCI; demolition / vegetation / water have captions but no change mask, so
# localization is reported for the two masked classes only (honest coverage).
QUERY_TO_MASK_CLASS: Dict[str, str] = {
    "new buildings or houses constructed": "building",
    "a new road or street built": "road",
}


class LevirMCIDataset(LevirCCDataset):
    """``LevirCCDataset`` plus pixel-level building/road change masks."""

    name = "levir_mci"

    def _mask_path(self, location_id: str) -> Path:
        r = self._records[location_id]
        return self.root / "images" / r["split"] / "label" / r["filename"]

    def has_mask(self, location_id: str) -> bool:
        return self._mask_path(location_id).exists()

    def load_change_mask(
        self,
        pair: PairKey,
        change_class: Optional[str] = None,
    ) -> np.ndarray:
        """Return the ground-truth change mask for *pair*.

        Args:
            pair: the bi-temporal pair to load the label for.
            change_class: ``"building"`` or ``"road"`` to get that class as a
                boolean mask; ``None`` to get the raw class-index map
                (0 background, 1 road, 2 building).

        Returns:
            ``[H, W]`` array. Boolean when *change_class* is given, else
            ``uint8`` class indices.
        """
        arr = np.array(Image.open(self._mask_path(pair.location_id)).convert("L"))
        if change_class is None:
            out = np.zeros_like(arr, dtype=np.uint8)
            out[arr == MASK_VALUE["road"]] = 1
            out[arr == MASK_VALUE["building"]] = 2
            return out
        if change_class not in MASK_VALUE:
            raise ValueError(
                f"unknown change_class {change_class!r}; expected one of "
                f"{sorted(MASK_VALUE)}")
        return arr == MASK_VALUE[change_class]
