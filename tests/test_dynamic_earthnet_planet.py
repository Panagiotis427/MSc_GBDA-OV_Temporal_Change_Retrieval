"""
Tests for the native-3 m Planet DynamicEarthNet loader (``dynamic_earthnet_planet``),
a registered dataset that previously had no test file.

Covers the pure, network-free pieces: the Planet->RGB composite, the one-hot->class
map collapse, the month-subsampling strategies, the zip-name indexing (regex parsing
only — no real GeoTIFF bytes needed), and the seeded, leakage-free train/val/test
split. Image/label *decoding* (rasterio) needs real data and is out of scope here.
"""
import zipfile

import numpy as np
import pytest

from src.datasets.dynamic_earthnet_planet import (
    DENPlanetDataset,
    build_index,
    onehot_to_classmap,
    planet_to_rgb,
    _select_months,
)


# ---------------------------------------------------------------------------
# planet_to_rgb
# ---------------------------------------------------------------------------
def test_planet_to_rgb_shape_dtype_and_range():
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    # Give the Red band (index 2) a ramp so the stretch has a real range.
    arr[..., 2] = np.arange(16, dtype=np.int16).reshape(4, 4)
    out = planet_to_rgb(arr, "rgb")
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_planet_to_rgb_rejects_unknown_bands():
    arr = np.zeros((2, 2, 4), dtype=np.int16)
    with pytest.raises(ValueError):
        planet_to_rgb(arr, "cmyk")


# ---------------------------------------------------------------------------
# onehot_to_classmap
# ---------------------------------------------------------------------------
def test_onehot_to_classmap_nodata_and_argmax():
    onehot = np.zeros((2, 2, 7), dtype=np.uint8)
    onehot[0, 0, :] = 0          # no channel set -> nodata (class 0)
    onehot[0, 1, 3] = 1          # channel 3 -> class index 4 (i + 1)
    onehot[1, 0, 0] = 1          # channel 0 -> class index 1
    onehot[1, 1, 6] = 1          # channel 6 -> class index 7
    cm = onehot_to_classmap(onehot)
    assert cm.shape == (2, 2)
    assert cm.dtype == np.uint8
    assert cm[0, 0] == 0
    assert cm[0, 1] == 4
    assert cm[1, 0] == 1
    assert cm[1, 1] == 7


# ---------------------------------------------------------------------------
# _select_months
# ---------------------------------------------------------------------------
def test_select_months_monthly_and_bimonthly():
    months = ["2018-01", "2018-02", "2018-03", "2018-04"]
    assert _select_months(months, "monthly") == months
    assert _select_months(months, "bimonthly") == ["2018-01", "2018-03"]


def test_select_months_seasonal_quartet_picks_one_per_season():
    months = ["2018-01", "2018-02", "2018-04", "2018-07", "2018-10", "2018-11"]
    got = _select_months(months, "seasonal-quartet")
    # earliest month of each present season, ordered winter->spring->summer->autumn
    assert got == ["2018-01", "2018-04", "2018-07", "2018-10"]


def test_select_months_unknown_strategy_raises():
    with pytest.raises(ValueError):
        _select_months(["2018-01"], "quarterly")


# ---------------------------------------------------------------------------
# _select_split — deterministic, disjoint, leakage-free
# ---------------------------------------------------------------------------
def test_select_split_none_and_all_return_everything():
    cubes = [f"c{i}" for i in range(10)]
    assert DENPlanetDataset._select_split(cubes, None, 0.2, 0.2, 42) == list(cubes)
    assert DENPlanetDataset._select_split(cubes, "all", 0.2, 0.2, 42) == list(cubes)


def test_select_split_invalid_raises():
    with pytest.raises(ValueError):
        DENPlanetDataset._select_split(["c0", "c1"], "holdout", 0.2, 0.2, 42)


def test_select_split_partition_is_disjoint_deterministic_and_complete():
    cubes = [f"c{i}" for i in range(10)]
    kw = dict(test_fraction=0.2, val_fraction=0.2, seed=42)
    train = DENPlanetDataset._select_split(cubes, "train", **kw)
    val = DENPlanetDataset._select_split(cubes, "val", **kw)
    test = DENPlanetDataset._select_split(cubes, "test", **kw)
    # disjoint buckets, together covering every cube exactly once
    assert set(train) & set(val) == set()
    assert set(train) & set(test) == set()
    assert set(val) & set(test) == set()
    assert set(train) | set(val) | set(test) == set(cubes)
    # deterministic: same seed reproduces the same partition
    assert DENPlanetDataset._select_split(cubes, "train", **kw) == train


# ---------------------------------------------------------------------------
# build_index — regex parsing over zip namelists (no TIFF bytes)
# ---------------------------------------------------------------------------
def _write_zip(path, names):
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, b"")  # names are all build_index reads


def test_build_index_keeps_only_cubes_with_two_plus_shared_months(tmp_path):
    cube, zone = "1700_3100_13", "13N"
    # cube A: 3 label months, 2 planet months -> 2 shared -> kept
    # cube B: 2 label months, 1 planet month  -> 1 shared -> dropped
    cube_b = "2235_3403_13"
    label_names = [
        f"labels/{cube}_{zone}/Labels/Raster/s/s-2018_01_01.tif",
        f"labels/{cube}_{zone}/Labels/Raster/s/s-2018_02_01.tif",
        f"labels/{cube}_{zone}/Labels/Raster/s/s-2018_03_01.tif",
        f"labels/{cube_b}_{zone}/Labels/Raster/s/s-2018_01_01.tif",
        f"labels/{cube_b}_{zone}/Labels/Raster/s/s-2018_02_01.tif",
    ]
    planet_names = [
        f"planet/{zone}/ll/{cube}/PF-SR/2018-01-01.tif",
        f"planet/{zone}/ll/{cube}/PF-SR/2018-02-01.tif",
        f"planet/{zone}/ll/{cube_b}/PF-SR/2018-01-01.tif",
    ]
    _write_zip(tmp_path / "labels.zip", label_names)
    _write_zip(tmp_path / f"planet.{zone}.zip", planet_names)

    index = build_index(str(tmp_path))
    assert set(index.keys()) == {zone}
    assert set(index[zone].keys()) == {cube}          # cube_b dropped (1 shared)
    entry = index[zone][cube]
    assert sorted(entry["label"]) == ["2018-01", "2018-02"]   # trimmed to shared
    assert sorted(entry["planet"]) == ["2018-01", "2018-02"]


def test_build_index_skips_zone_without_planet_zip(tmp_path):
    zone = "13N"
    _write_zip(tmp_path / "labels.zip", [
        f"labels/1700_3100_13_{zone}/Labels/Raster/s/s-2018_01_01.tif",
        f"labels/1700_3100_13_{zone}/Labels/Raster/s/s-2018_02_01.tif",
    ])
    # no planet.13N.zip on disk -> the whole zone is skipped, empty index
    index = build_index(str(tmp_path))
    assert index == {}
