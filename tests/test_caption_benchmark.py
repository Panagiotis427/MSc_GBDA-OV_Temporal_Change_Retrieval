"""
Guards that the shared caption-benchmark driver keys its embedding cache through
``cache_tag_for`` rather than a raw split string. For the RGB-only LEVIR-CC /
SECOND-CC datasets the tag *value* is identical either way, so a value check
can't catch a regression — instead we spy on ``cache_tag_for`` and assert the
driver actually routes through it (a revert to ``cache_tag=args.split`` would not
call it), and that the tag it produces reaches ``load_or_compute``.
"""
from unittest import mock

import scripts._caption_benchmark as cb
from src.embeddings import cache_tag_for


class _Label:
    change_type = "building"


class _Query:
    text = "new buildings"

    def predicate(self, lb):
        return True


class _QR:
    text = "new buildings"
    ap = 0.5
    n_relevant = 1


class _Report:
    approach = "zero_shot"
    mAP = 0.5
    per_query = [_QR()]


def test_driver_keys_cache_via_cache_tag_for(tmp_path):
    ds = mock.Mock()
    ds.list_pairs.return_value = [object()]
    ds.get_pair_label.return_value = _Label()
    captured = {}

    def _fake_load_or_compute(dataset, enc, cache_dir, cache_tag):
        captured["tag"] = cache_tag
        return mock.Mock()

    with mock.patch.object(cb, "build_dataset", return_value=ds), \
         mock.patch.object(cb, "get_encoder", return_value=mock.Mock()), \
         mock.patch.object(cb, "get_queries", return_value=[_Query()]), \
         mock.patch.object(cb, "ChangeRetriever", return_value=mock.Mock()), \
         mock.patch.object(cb, "run_benchmark", return_value=_Report()), \
         mock.patch.object(cb, "load_or_compute", side_effect=_fake_load_or_compute), \
         mock.patch.object(cb, "cache_tag_for", wraps=cache_tag_for) as spy_tag:
        cb.run_caption_benchmark(
            "levir_cc", display_name="LEVIR-CC", default_root=str(tmp_path),
            root_help="x",
            argv=["--split", "test", "--encoders", "clip_vitl14",
                  "--cache-dir", str(tmp_path), "--results-dir", str(tmp_path)],
        )

    spy_tag.assert_called_once_with("test", "rgb")          # routed through the helper
    assert captured["tag"] == cache_tag_for("test", "rgb")  # that value reached the cache


def test_cache_tag_for_would_distinguish_colour():
    # Why the spy matters: for rgb the tag equals the raw split, so only the fact
    # that cache_tag_for is *called* protects a future colour variant from colliding.
    assert cache_tag_for("test", "rgb") != cache_tag_for("test", "nrg")
