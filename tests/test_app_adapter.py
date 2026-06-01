"""Regression: SemanticChangeSearch._maybe_load_adapter must respect color_mode.

Bug: it loaded models/<ds>__<enc>__adapter.pt regardless of colour, so in NRG
mode it silently applied the RGB-trained adapter to NRG embeddings. Fix tags the
filename with the colour, matching run_pipeline's save convention.
"""
from __future__ import annotations

from pathlib import Path

from src.app import RunConfig, SemanticChangeSearch
from src.model import create_projection_head, save_adapter


def _save_rgb_adapter(models_dir: Path, ds="dynamic_earthnet", enc="clip_vitl14"):
    models_dir.mkdir(parents=True, exist_ok=True)
    head = create_projection_head(input_dim=8, output_dim=8, hidden_dims=(16,))
    save_adapter(str(models_dir / f"{ds}__{enc}__adapter.pt"), head, {
        "input_dim": 8, "output_dim": 8, "hidden_dims": [16],
        "dropout_rate": 0.3, "feature_mode": "difference",
    })


def _cfg(tmp_path, color_mode):
    # path = Path(cache_dir).parent / "models"  ->  tmp/models
    return RunConfig(dataset="dynamic_earthnet", encoder="clip_vitl14",
                     cache_dir=str(tmp_path / "cache"), color_mode=color_mode)


def test_rgb_loads_rgb_adapter(tmp_path):
    _save_rgb_adapter(tmp_path / "models")
    engine = SemanticChangeSearch.__new__(SemanticChangeSearch)  # skip _build
    assert engine._maybe_load_adapter(_cfg(tmp_path, "rgb")) is not None


def test_nrg_does_not_load_rgb_adapter(tmp_path):
    # Only an RGB adapter exists; NRG mode must NOT pick it up.
    _save_rgb_adapter(tmp_path / "models")
    engine = SemanticChangeSearch.__new__(SemanticChangeSearch)
    assert engine._maybe_load_adapter(_cfg(tmp_path, "nrg")) is None


def test_difference_mode_ignores_mode_tagged_adapter(tmp_path):
    # run_pipeline saves concatenate adapters as ..._concatenate__adapter.pt;
    # the app (difference, default) must NOT load a mode-tagged adapter.
    # Use a fake encoder name so the _PROJECT_ROOT/models fallback (which holds
    # the real clip_vitl14 adapter) can't accidentally satisfy the lookup.
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    head = create_projection_head(input_dim=8, output_dim=8, hidden_dims=(16,))
    save_adapter(str(models / "dynamic_earthnet__zzmock_concatenate__adapter.pt"),
                 head, {"input_dim": 8, "output_dim": 8, "hidden_dims": [16],
                        "feature_mode": "concatenate"})
    cfg = RunConfig(dataset="dynamic_earthnet", encoder="zzmock",
                    cache_dir=str(tmp_path / "cache"), color_mode="rgb")
    engine = SemanticChangeSearch.__new__(SemanticChangeSearch)
    # no plain ..._adapter.pt present -> difference-mode lookup returns None
    assert engine._maybe_load_adapter(cfg) is None
