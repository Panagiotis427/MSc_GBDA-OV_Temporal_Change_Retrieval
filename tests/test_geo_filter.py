"""Tests for src/geo_filter.py — geographic region and bbox filtering."""
from __future__ import annotations

import json

import pytest

from src.datasets.base import PairKey
from src.geo_filter import GeoFilter, REGION_ALL

_META = {
    "aoi_eu": {"lat_c": 48.0, "lon_c": 2.0, "region": "Europe"},
    "aoi_na": {"lat_c": 40.7, "lon_c": -74.0, "region": "North America"},
    "aoi_as": {"lat_c": 35.7, "lon_c": 139.7, "region": "Asia"},
}

_PAIRS = [
    PairKey("aoi_eu", "t1", "t2"),
    PairKey("aoi_na", "t1", "t2"),
    PairKey("aoi_as", "t1", "t2"),
    PairKey("aoi_unknown", "t1", "t2"),  # not in metadata
]


@pytest.fixture
def gf(tmp_path):
    p = tmp_path / "aoi_metadata.json"
    p.write_text(json.dumps(_META))
    return GeoFilter(p)


# -- regions ----------------------------------------------------------------

def test_regions_includes_all(gf):
    r = gf.regions
    assert r[0] == REGION_ALL


def test_regions_sorted(gf):
    r = gf.regions[1:]  # skip "All"
    assert r == sorted(r)


def test_regions_all_present(gf):
    r = gf.regions
    assert "Europe" in r
    assert "North America" in r
    assert "Asia" in r


# -- filter_by_region -------------------------------------------------------

def test_filter_all_returns_unchanged(gf):
    assert gf.filter_by_region(_PAIRS, REGION_ALL) == _PAIRS


def test_filter_empty_string_returns_unchanged(gf):
    assert gf.filter_by_region(_PAIRS, "") == _PAIRS


def test_filter_europe(gf):
    result = gf.filter_by_region(_PAIRS, "Europe")
    assert len(result) == 1
    assert result[0].location_id == "aoi_eu"


def test_filter_north_america(gf):
    result = gf.filter_by_region(_PAIRS, "North America")
    assert len(result) == 1
    assert result[0].location_id == "aoi_na"


def test_filter_excludes_unknown_location(gf):
    result = gf.filter_by_region(_PAIRS, "Europe")
    loc_ids = [p.location_id for p in result]
    assert "aoi_unknown" not in loc_ids


def test_filter_nonexistent_region_returns_empty(gf):
    result = gf.filter_by_region(_PAIRS, "Antarctica")
    assert result == []


# -- filter_by_bbox ---------------------------------------------------------

def test_bbox_europe(gf):
    result = gf.filter_by_bbox(_PAIRS, lat_min=40, lat_max=55, lon_min=-5, lon_max=15)
    loc_ids = [p.location_id for p in result]
    assert "aoi_eu" in loc_ids
    assert "aoi_na" not in loc_ids
    assert "aoi_as" not in loc_ids
    # unknown location kept (benefit of the doubt)
    assert "aoi_unknown" in loc_ids


# -- centroid ---------------------------------------------------------------

def test_centroid_known(gf):
    c = gf.centroid("aoi_eu")
    assert c is not None
    lat, lon = c
    assert abs(lat - 48.0) < 1e-6
    assert abs(lon - 2.0) < 1e-6


def test_centroid_unknown(gf):
    assert gf.centroid("aoi_unknown") is None


# -- nearest ----------------------------------------------------------------

def test_nearest_returns_sorted_ascending(gf):
    near = gf.nearest("aoi_eu", ["aoi_na", "aoi_as", "aoi_eu"])
    dists = [d for _, d in near]
    assert dists == sorted(dists)


def test_nearest_excludes_self(gf):
    near = gf.nearest("aoi_eu", ["aoi_eu", "aoi_na"])
    assert all(loc != "aoi_eu" for loc, _ in near)
