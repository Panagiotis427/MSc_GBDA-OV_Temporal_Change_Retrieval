"""
Locks the adapter-filename contract between the producer (``scripts.run_pipeline``,
which saves ``models/<ds>__<enc>[_<color>][_<split>][_<mode>]__adapter.pt``) and the
consumer (``scripts.export_results.adapter_path``, which must resolve the same path).

The default (train split + difference mode) and colour-only names are asserted
against filenames actually committed under ``models/``; the non-default split/mode
cases guard against the regression where ``adapter_path`` ignored the split/mode
tags and silently reported "no adapter" for a validly-trained non-default run.
"""
from scripts.export_results import adapter_path


def test_default_name_matches_committed_artifact():
    # models/dynamic_earthnet__clip_vitl14__adapter.pt is committed.
    assert (adapter_path("dynamic_earthnet", "clip_vitl14", "rgb").name
            == "dynamic_earthnet__clip_vitl14__adapter.pt")


def test_colour_suffix_matches_committed_artifact():
    # models/dynamic_earthnet__clip_vitl14_concatenate__adapter.pt is committed;
    # here we check the colour suffix alone (nrg) uses a single underscore.
    assert (adapter_path("dynamic_earthnet", "georsclip", "nrg").name
            == "dynamic_earthnet__georsclip_nrg__adapter.pt")


def test_concatenate_mode_matches_committed_artifact():
    # models/dynamic_earthnet__clip_vitl14_concatenate__adapter.pt is committed.
    assert (adapter_path("dynamic_earthnet", "clip_vitl14", "rgb",
                         mode="concatenate").name
            == "dynamic_earthnet__clip_vitl14_concatenate__adapter.pt")


def test_non_default_split_and_mode_are_both_tagged():
    # The producer names a `--train-split val --mode concatenate` run this way;
    # the consumer must resolve the identical path (order: color, split, mode).
    assert (adapter_path("dynamic_earthnet", "georsclip", "nrg",
                         train_split="val", mode="concatenate").name
            == "dynamic_earthnet__georsclip_nrg_val_concatenate__adapter.pt")


def test_non_default_split_only():
    assert (adapter_path("dynamic_earthnet", "clip_vitl14", "rgb",
                         train_split="val").name
            == "dynamic_earthnet__clip_vitl14_val__adapter.pt")
