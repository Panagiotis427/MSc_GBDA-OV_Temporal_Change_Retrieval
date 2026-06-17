# Architecture & contribution contract

The pipeline is **dataset-agnostic**. Every shared file (`embeddings.py`,
`retrieval.py`, `benchmark.py`, `train.py`, `app.py`,
`scripts/run_pipeline.py`) consumes only the `TemporalDataset` protocol and
the dataset/encoder/query registries. Concrete loaders and dataset-specific
choices live in their own modules and self-register.

**Hard rule: adding a dataset = adding files only, never editing shared pipeline files.**

## Plug-in points

| Concern | Where | How |
|---|---|---|
| Dataset loader | `src/datasets/<name>.py` | Implement the `TemporalDataset` protocol (see `src/datasets/base.py`) |
| Loader registration | same file | Call `register_dataset(name, factory, opts_adapter)` from `src/datasets/registry.py`, then add one line to `src/datasets/registry.py` *only* if a new built-in is intended — third-party datasets can register from their own module on import |
| Generic-options mapping | same opts adapter | Maps `(root, pairing, split, **extra)` -> loader kwargs; `color_mode` travels via `**extra` and is forwarded to `DENNpyDataset` |
| Encoder | `src/encoders/<name>.py` | Implement `ImageTextEncoder`; `register_encoder(...)` in `src/encoders/__init__.py` |
| Benchmark query set | `src/queries/<name>.py` | List of `Query(text, category, predicate)`; `register_queries(name, queries)`; imported automatically by `src/queries/__init__.py` |
| App / CLI | nothing — dropdown choices and `--dataset` / `--encoder` choices are derived from the registries |

## What a new dataset should add (and only add)

```
src/datasets/<name>.py          # loader, register_dataset(...)
src/queries/<name>.py           # query set, register_queries(...)
src/queries/__init__.py         # ONE import line: `from . import <name>`
tests/test_<name>.py            # loader-level tests (e.g. test_second_cc.py)
```

If you find yourself editing `embeddings.py`, `retrieval.py`, `benchmark.py`,
`train.py`, `app.py`, or `scripts/run_pipeline.py` for a new dataset, stop —
an existing extension point already covers it.

## Quick recipe — adding QFabric (worked example)

1. Implement `TEOChatlasQFabricDataset(TemporalDataset)` in `src/datasets/qfabric_teo.py`
   (already present). `get_pair_label` must return a `PairLabel` for the
   quantitative benchmark to work; without it only qualitative retrieval runs.
2. Register: `register_dataset("qfabric_teo", _qfabric_teo_factory, _qfabric_teo_opts)`
   (already done in `src/datasets/registry.py`; opts adapter maps `root` to
   the crop directory).
3. Add `src/queries/qfabric.py` with `register_queries("qfabric_teo", QUERIES)`;
   one-line import in `src/queries/__init__.py`.
4. Run: `python -m scripts.benchmark_qfabric --root <dir> --encoder clip_vitl14`

## Cache and artefact paths

- Embeddings: `data/cache/<dataset>__<encoder>[__<tag>]__pair_embeddings.npz`
  where `<tag>` = `{split}[_{color_mode}][_lora]` (e.g. `train`, `test_nrg`,
  `train_nrg_lora`) — built by `cache_tag_for()`. Pass `cache_tag` to
  `load_or_compute()` to isolate caches per split/colour/LoRA. Without a tag the
  legacy `<dataset>__<encoder>__pair_embeddings.npz` path is used
  (backwards-compatible with old test-split caches).
- Adapters: `models/<dataset>__<encoder>[__<color>][__<split>][__<mode>]__adapter.pt`
  — the committed `train` split and `difference` feature mode take **no** suffix
  (back-compat); a non-default `--train-split` appends `_<split>` and any other
  feature mode appends `_<mode>`. `train.py` and `run_pipeline.py` share this
  convention.
- Keyed by `(dataset, encoder, split, color_mode)` — no collision across splits or colour modes.
- The pair-set is validated on cache load; a stale pair-set triggers automatic recompute and overwrites the cache at the same path.

## Adapters

Trained adapters in `models/` are committed (~3 MB each, keyed by
`(dataset, encoder[, color])`). Retrain with `scripts/run_pipeline.py`
only if encoders or supervision change.

## Shared helpers (not plug-in points)

A few small modules are shared *infrastructure*, not part of the
dataset-agnostic contract above — they are imported freely and have no
registry:

- `src/stats.py` — `rand_ap(...)`, the shuffle-based random-AP baseline used by
  the significance scripts (`scripts/significance_audit.py`, `scripts/patch_eval.py`).
  `scripts/cv_eval.py` keeps its own `rng.permutation`-based variant on purpose,
  to avoid perturbing its committed RNG-dependent results.
- `src/embeddings.py::cache_tag_for(split, color_mode, lora)` — the single source
  of truth for split/colour/LoRA cache tags; import it rather than re-deriving tags.

## Dependencies

`pyproject.toml` is the full dev install (`pip install -e .`: runtime + pytest/coverage/
mlflow/faiss + `opencv-python`); `requirements.txt` is the lightweight HuggingFace-Space
subset (no dev tools, `opencv-python-headless`). See the README "Dependencies" section.
