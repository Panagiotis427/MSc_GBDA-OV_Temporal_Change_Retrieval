"""
Shared matplotlib figure-save helper for the ``scripts.make_*`` figure
generators.

Kept dependency-light (matplotlib only, imported lazily inside the function so
this module never forces a backend before the caller has selected one) and
torch-free, matching the "figure scripts are pure result consumers" convention.
"""
from __future__ import annotations

from pathlib import Path


def save_fig(fig, out_dir, name: str, svg: bool = False, dpi: int = 150) -> Path:
    """Save *fig* as ``<out_dir>/<name>.png`` (and ``.svg`` when *svg*), close it,
    and return the PNG path.

    Single source of truth for the figure scripts' save convention (150 dpi,
    tight bbox) so a change to that convention lands in one place.
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if svg:
        fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    return path
