---
title: Open Vocabulary Temporal Change Retrieval
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.14.0"
app_file: app.py
pinned: false
---

# Open Vocabulary Temporal Change Retrieval (GBDA Lab Project)

A *Semantic Change Search Engine*: given a natural-language query
(e.g. *"new buildings on former agricultural land"*, *"forest cleared to bare
soil"*), retrieve the satellite image **pairs and the timestep** where that
change occurred — across a multitemporal dataset, without training a
class-specific detector.

Frozen vision-language backbones (CLIP / GeoRSCLIP / RemoteCLIP) encode each
timestep; a bi-temporal *change feature* is matched against the query text.
Primary dataset: **Dynamic EarthNet (DEN)**; the abstraction is
dataset-agnostic (QFabric / fMoW slot in via the registry).

> **Just want to run the app?** Jump to [Run / install / use](#run--install--use) —
> install, then a 30-second synthetic demo or the real dataset.

## Demo

![Gradio UI — semantic change search engine](assets/app_screenshot.png)

*Enter a free-text change query, pick a dataset / encoder / scoring approach, and get
ranked before→after pairs with a change heatmap on T2.* To (re)generate the screenshot
locally (it lands in `assets/app_screenshot.png`):

```bash
pip install -e .
python -m scripts.make_den_fixture
python -m src.app --root tests/fixtures/den_tiny --split all --encoder clip_vitl14   # http://127.0.0.1:7860
# then save a screenshot of the browser tab to assets/app_screenshot.png
```

## Pipeline

End-to-end flow for one user query, with the module responsible at each
step:

```
┌─ offline (one-time per dataset+encoder, cached) ─────────────────────────┐
│                                                                          │
│  TemporalDataset.list_pairs()                src/datasets/*               │
│           │                                                               │
│           ▼                                                               │
│  load_pair_images(pair)  ──▶  PIL T1, T2                                  │
│           │                                                               │
│           ▼                                                               │
│  ImageTextEncoder.encode_image     src/encoders/*  (CLIP / GeoRS /        │
│           │                                          RemoteCLIP, frozen)  │
│           ▼                                                               │
│  f_T1, f_T2  (L2-normed, [N, D])  ──cache──▶                              │
│   data/cache/<dataset>__<encoder>__<split>[_<color>][_lora]__pair_embeddings.npz │
│                                       src/embeddings.py                   │
└──────────────────────────────────────────────────────────────────────────┘

┌─ adapter training (only for `peft` approach; offline) ───────────────────┐
│                                                                          │
│  weak caption per pair  ──ProjectionHead──▶  masked symm. InfoNCE         │
│  (e.g. "agriculture replaced by impervious surface")                     │
│                                              src/train.py                 │
│  → models/<dataset>__<encoder>__adapter.pt                                │
└──────────────────────────────────────────────────────────────────────────┘

┌─ inference (per query, hot path) ────────────────────────────────────────┐
│                                                                          │
│  user query text  ─encoder.encode_text─▶  t  (same shared [D] space)     │
│                                                                          │
│       ┌───────────────── ChangeRetriever.score_all ──────────────────┐    │
│       │   naive      :  t · f_T2                                     │    │
│       │   zero_shot  :  t · f_T2  −  t · f_T1   (Δ-similarity)       │    │
│       │   peft       :  t · g(Δf)   with Δf = f_T2 − f_T1            │    │
│       └────────────────────────────────────────────────────────────┘    │
│                              src/retrieval.py                            │
│                                    │                                     │
│                                    ▼                                     │
│  rank all pairs by score  ──▶  top-K change events                       │
│                                    │                                     │
│                                    ▼                                     │
│  for top-1: dataset.load_pair_images(pair)  +                            │
│  encoder.compute_patch_text_similarity → heatmap on T2                   │
│                                              src/app.py + src/heatmap.py │
│                                                                          │
│  label-grounded benchmark (offline, optional)                            │
│  per-query relevance from PairLabel → Recall@K, mAP, seasonal drift      │
│                                              src/benchmark.py            │
└──────────────────────────────────────────────────────────────────────────┘
```

Three scoring **approaches** (the supervisor-requested comparison):

| Approach    | Score                                       | Training |
|-------------|---------------------------------------------|----------|
| `naive`     | cos(t, f_T2)                                | none (lower bound) |
| `zero_shot` | cos(t, f_T2) − cos(t, f_T1)  (Δ-similarity) | none |
| `peft`      | cos(t, g(Δf)), g = trained ProjectionHead   | ~0.5–0.7 M params (adapter only; backbones frozen) |

*Per-encoder results for all three approaches — in-distribution and cross-split — are in [`REPORT.md`](REPORT.md) §7.*

**Key decoupling:** `f_T1, f_T2` are cached per `(dataset, encoder, split, color_mode)` so all
three approaches and any number of queries reuse the same one-time encode
pass. Adding a dataset = implementing the `TemporalDataset` protocol +
registering — the entire flow above re-uses the new dataset unchanged
(see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)).

**Evaluation** is label-grounded: a fixed query set (per dataset, under
`src/queries/<name>.py`) maps each query to a relevance rule over the
derived `PairLabel`s → Recall@K, mAP, plus a seasonal-vs-permanent
("semantic drift") error report.

## Module map

| File | Role |
|------|------|
| `src/datasets/` | `TemporalDataset` protocol, `DENDataset` (raster), `DENNpyDataset` (DynNet `.npy` + `color_mode` rgb/nrg/ndvi via NIR infrared frames), `QFabricDataset` (`images_only` parquet), `TEOChatlasQFabricDataset` (`qfabric_teo` — QFabric crops + real change-type labels), layout-detecting registry + opts adapters |
| `src/queries/` | Per-dataset query sets (`den.py`, `qfabric.py`); registry resolved by `dataset.name` |
| `src/results_io.py` | serialize `BenchmarkReport` to JSON/CSV (torch-free); consumed by the figure scripts |
| `src/error_analysis.py` | per-query confusion matrix + precision/recall (seasonal-vs-permanent error analysis) |
| `src/encoders/` | `ImageTextEncoder` protocol; `clip_vitl14` (768-d), `georsclip` (512-d), `remoteclip` (768-d) |
| `src/text_encoder.py` | frozen CLIP text tower (`text_model` + `text_projection`, device-aware) |
| `src/features.py` | `compute_change_feature` (difference / concatenate) |
| `src/embeddings.py` | per-pair `f_T1,f_T2` compute + npz cache (`PairEmbeddingStore`); `cache_tag` arg keys cache by split+color to prevent cross-split collision |
| `src/retrieval.py` | `ChangeRetriever` — naive / zero_shot / peft scoring |
| `src/benchmark.py` | query set + label relevance, Recall@K / mAP / drift |
| `src/model.py` | `ProjectionHead` adapter, InfoNCE, adapter save/load |
| `src/train.py` | PEFT training (masked symmetric InfoNCE on weak captions) |
| `src/lora_train.py` | LoRA fine-tuning of visual encoder via peft; `train_lora`, `merge_lora_into_encoder`, `save_lora` |
| `src/geo_filter.py` | `GeoFilter` — filter pairs by continental region or lat/lon bbox using `aoi_metadata.json`; toggleable |
| `src/rerank.py` | `Reranker` — post-retrieval re-ranking: `diversity` (unique AOIs) or `coherence` (cluster near top-1); toggleable |
| `src/app.py` | Gradio engine + UI (Dataset / Encoder / Approach selectors) |
| `app.py` | HuggingFace Spaces entry point (uses tiny fixture by default; override via env vars) |
| `scripts/download_den.py` | fetch + extract DEN subset, build label index |
| `scripts/download_qfabric.py` | fetch a QFabric image subset from HuggingFace (images-only; qualitative demo) |
| `scripts/build_qfabric_labels.py` | TEOChatlas RQA2 → `qfabric_teo_labels.json` (27,879 real crop→change-type labels) |
| `scripts/benchmark_qfabric.py` | extract QFabric crops + encode + label-grounded change-type mAP (`qfabric_teo`) |
| `scripts/make_den_fixture.py` | tiny synthetic DEN tree for fast tests |
| `scripts/run_pipeline.py` | one-command run with `--train-split` / `--eval-splits` / `--color-mode` / `--mode` / `--lora` / `--results-dir`; cross-split mAP table |
| `scripts/export_results.py` | regenerate benchmarks from cache → `results/*.json` + `macro_summary.csv` (`--confusion` for error analysis) |
| `scripts/make_figures.py` | publication PNGs (recall curves, mAP bars, colour heatmap, seasonal drift, cross-split, confusion) from `results/` |
| `scripts/make_comparison_figure.py` | static zero-shot-vs-PEFT top-K visual comparison per encoder |
| `scripts/lora_sweep.py` | LoRA rank/epoch sweep (georsclip+nrg), in-memory, no cache/model clobber |
| `scripts/significance_audit.py` | random-ranking baseline + permutation p + BH-FDR over every result cell → `results/results_audit_summary.csv` (REPORT Appendix B) |
| `scripts/cv_eval.py` | full-corpus + k-fold AOI cross-validation with bootstrap CIs; `--relevance fraction` swaps dominant-class-flip relevance for pixel-fraction (REPORT Appendix B.8–B.9); merges cached split embeddings, no re-encode |
| `scripts/patch_eval.py` | patch-level (localised) Δ-similarity change retrieval vs the global baseline (REPORT Appendix B.10, "S3"); caches per-patch embeddings via `encoder.encode_image_patches`. `--approach hybrid` fuses global+patch (B.11), `patch_softattn`/`patch_spatial` are training-free change-attention variants (B.12), `--prompt-ensemble` averages query templates (B.11) |

## Run / install / use

**Requirements:** Python 3.12+ · ~3 GB disk (model weights) · ~9 GB more for real DEN · GPU optional.
RS-encoder weights download from HuggingFace on first use into `.model_cache/`. On-disk DEN
layouts (`planet/*.tif` raster or DynNet preprocessed `.npy`) are auto-detected; dataset sources
in [`docs/Common_Resources.md`](docs/Common_Resources.md).

### 1. Setup (one-time)

```bash
git clone <repo-url> && cd MSc_GBDA-OV_Temporal_Change_Retrieval

python -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
                                   #   if blocked once: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

pip install -e .
```

### 2. Option A — 30-second synthetic demo (no download)

```bash
python -m scripts.make_den_fixture
# Builds tests/fixtures/den_tiny/: 2 AOIs × 8 months, <1 MB, deterministic.

python -m src.app --root tests/fixtures/den_tiny --split all --encoder clip_vitl14
# First run downloads CLIP weights (~1.6 GB) into .model_cache/ — one-time.
# Open http://127.0.0.1:7860
```

### 2. Option B — real Dynamic EarthNet (~7 GB)

```bash
python -m scripts.download_den --dest data/DynamicEarthNet
# ~7 GB ZIP via gdown; extracted; idempotent (_done.marker guards re-runs).

python -m src.app --root data/DynamicEarthNet --encoder clip_vitl14
# Defaults: --split train (55 AOIs, 605 pairs), --approach zero_shot.
# Switch to --approach peft in the UI for the trained-adapter scoring.
```

### App usage

Enter a query, press **Search**. Example queries: `agricultural land converted to wetland` ·
`new buildings on former farmland` · `forest cleared to bare soil`. Results: T1 / T2 tiles side
by side · heatmap on T2 · confidence (0–1) · permanence note (`permanent` / `likely SEASONAL` /
`stable`) · ranked table. Two control accordions: **Settings** (Dataset / Encoder / Approach —
naive / zero-shot / **patch** (localised, best on DEN) / PEFT — / Color Mode / LoRA — needs
**Apply** to rebuild embeddings) and **Filters & Re-ranking**
(geographic filter, re-ranking — next **Search**, no Apply). Startup defaults via CLI flags:

| Flag | Default | Notes |
|---|---|---|
| `--root` | `data/DynamicEarthNet` | Path to dataset; DEN layout auto-detected. |
| `--split` | `train` | DEN AOI split: `train` (605 pairs), `val`/`test` (110 each), `all` (825). |
| `--pairing` | `bimonthly` | How DEN's 24 monthly timesteps pair into (T1, T2). |
| `--port` | `7860` | Gradio HTTP port (in use? add `--port 7861`). |
| `--color-mode` | `rgb` | `rgb` / `nrg` (NIR-Red-Green, best zero-shot with GeoRSCLIP) / `ndvi`. |
| `--lora` / `--no-lora` | off | Load LoRA-adapted embeddings (pre-cache via `run_pipeline --lora`). |
| `--geo-filter` / `--no-geo-filter` | off | Geographic region filter. |
| `--rerank` / `--no-rerank` | off | Post-retrieval re-ranking. |
| `--rerank-strategy` | `diversity` | `diversity` = unique AOIs; `coherence` = cluster near top-1. |

> `peft` errors "no adapter" → adapter missing from `models/`; train with `run_pipeline` (below)
> or switch to `zero_shot`. **Hosted demo (no install):** push the repo to a HuggingFace Space —
> `app.py` + `requirements.txt` are ready (see [`docs/EXTENSIONS.md`](docs/EXTENSIONS.md)).

### Developer — pipeline, training, tests

```bash
# Full pipeline: train on train split, evaluate on all three splits
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder clip_vitl14 --train-split train --eval-splits train val test --epochs 40

# Best zero-shot generalisation (GeoRSCLIP + NIR, no training)
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder georsclip --color-mode nrg --eval-splits train val test --skip-train

# LoRA adapter on the visual encoder
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder georsclip --color-mode nrg --skip-train \
    --lora --lora-epochs 20 --lora-rank 4 --lora-alpha 8 --eval-splits train val test
```

Repeat with `--encoder georsclip` / `remoteclip` for the three-encoder comparison in
[`REPORT.md`](REPORT.md) §7. `run_pipeline` is the canonical, cache-consistent flow; the
individual stages (`src.embeddings`, `src.benchmark`, `src.train`, `src.lora_train`) are
convenience entry points — pass the same `--split` / `--color-mode` to every stage so they
share the split-tagged embedding cache.

```bash
pytest -q                              # full suite: 212 passed, 1 skipped (incl. real-CLIP test_text_encoder, needs weights, ~45 s)
pytest -q --ignore=tests/test_text_encoder.py   # skip the real-CLIP-weights test (~45 s) for the fast CPU loop
```

## Extending

Adding a dataset is file-additive only — never edit shared pipeline files.
Full contract: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
