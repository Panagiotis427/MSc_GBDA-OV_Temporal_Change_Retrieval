"""
Tests for `cache_tag_for` — the single source of truth for the embedding-cache
tag (`<split>[_<color>][_lora]`). The docstring notes a historical
``test``+``rgb`` -> empty-tag drift bug; this locks the tag string so split/colour
caches never collide again.
"""
from src.embeddings import cache_tag_for


def test_cache_tag_rgb_adds_no_colour_suffix():
    assert cache_tag_for("train") == "train"
    assert cache_tag_for("test", "rgb") == "test"
    assert cache_tag_for("val", color_mode="rgb") == "val"


def test_cache_tag_colour_modes():
    assert cache_tag_for("val", "nrg") == "val_nrg"
    assert cache_tag_for("train", "ndvi") == "train_ndvi"


def test_cache_tag_lora_suffix():
    assert cache_tag_for("test", "rgb", lora=True) == "test_lora"
    assert cache_tag_for("train", "nrg", lora=True) == "train_nrg_lora"


def test_cache_tags_do_not_collide_across_splits_and_colours():
    tags = {
        cache_tag_for("train", "rgb"),
        cache_tag_for("val", "rgb"),
        cache_tag_for("test", "rgb"),
        cache_tag_for("train", "nrg"),
        cache_tag_for("train", "ndvi"),
        cache_tag_for("train", "rgb", lora=True),
    }
    assert len(tags) == 6  # all distinct
