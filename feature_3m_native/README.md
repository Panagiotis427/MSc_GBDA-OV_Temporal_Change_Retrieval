# Native 3 m Planet-Fusion DEN

Change-retrieval evaluation run **directly on the native 3 m Planet-Fusion
imagery** of DynamicEarthNet — the source that the report's §Data had **excluded**
in favour of the ~7 GB preprocessed RGB/NIR-JPEG subset.

**Question:** does working on the lossy JPEG subset *cost* retrieval performance?
**Answer: no** — the native rasters land in the same band, if anything slightly
better, so the data-source choice is vindicated.

## Motivation — the excluded source

All DEN results in the report run on either the preprocessed RGB/NIR-**JPEG**
subset (~7 GB, lossy) or downsampled composites. The report's §Data deliberately
**excluded** the full ~525 GB raw TUM mirror. That left one question open: *are we
losing accuracy because we evaluate on compressed / resampled imagery rather than
the full-quality source?*

Answering it requires evaluating on the **native Planet-Fusion surface-reflectance
rasters** — full radiometric depth, no JPEG loss.

## What was added (and where)

**Architecture integration** (additive — does not change any existing dataset):
- `src/datasets/dynamic_earthnet_planet.py` — the `DENPlanetDataset` loader. Reads
  the PF-SR `int16` files (1024×1024×**4**, Blue/Green/Red/NIR, 3 m) **straight from
  the compressed `planet.<UTM>.zip` archives** via `rasterio` MemoryFile (no lossy
  re-encoding, no need to unpack 525 GB), converts the 7-channel one-hot masks to a
  class-index map and reuses DEN's `derive_pair_label`.
- `src/datasets/registry.py`, `src/queries/den.py` — two registration lines
  (`dynamic_earthnet_planet`). This is the registry's **intended extension point**;
  adding a dataset name cannot break the existing loaders.
- `scripts/download_den_planet.py` — `rsync` cherry-pick of **only** the needed UTM
  zones + `labels.zip` (instead of the full mirror).

**Evaluation** (self-contained — does **not** touch `scripts/cv_eval.py`):
- `feature_3m_native/cv_eval_planet3m.py` — 5-fold AOI cross-validation (zero-shot +
  leakage-free PEFT), plus a full-corpus AP estimate with bootstrap CIs and
  permutation p-values. It mirrors the logic of `scripts/cv_eval.py` so the numbers
  are **directly comparable**, while importing the `src/` library read-only.
- `feature_3m_native/results/` — the output JSON.

## How to run

```bash
# 1) (once) encode every cube into a single cache — needs the disk holding the zips
uv run python -m src.embeddings --dataset dynamic_earthnet_planet \
  --root /path/to/dir/with/planet.<UTM>.zip --split all --color-mode rgb

# 2) cross-validation (zero-shot + PEFT)
uv run python feature_3m_native/cv_eval_planet3m.py \
  --root /path/to/dir/with/planet.<UTM>.zip --folds 5 --peft
```

## Result (CLIP ViT-L/14, 23 AOIs, 253 pairs, 5-fold AOI CV, dominant relevance)

| Source | colour | macro mAP (zero-shot) | corpus |
|--------|--------|------------------------|--------|
| **Native 3 m raster** | RGB | **0.130 ± 0.068** | 23 AOIs |
| JPEG subset (committed) | NRG | 0.076 | 75 AOIs |
| JPEG subset (fraction relevance) | NRG | 0.123 | 75 AOIs |

The comparison is not fully controlled (the corpora differ in AOI count, 23 vs 75,
and colour composite, RGB vs NRG). But colour barely moves the score on this corpus
(georsclip on the JPEG subset: RGB 0.085 vs NRG 0.100), so the direction is robust:
**the native source is no worse.**

> The leakage-free k-fold PEFT estimate is 0.215 ± 0.306, but it is **unstable** (one
> fold at 0.75, the rest 0.03–0.19, driven by few positives per small fold), so the
> zero-shot figure is the reliable point of comparison. Only **5/10** queries are
> evaluable — the others have no positives across the 23 AOIs, the same
> data-coverage limit the report notes for the JPEG corpus.

## How this connects to the wider conclusions

This is one more structurally different attempt at the same ceiling. The report
already shows the bottleneck is not the spectral channels (NIR), not the encoder
(RemoteCLIP), and not spatial tiling (patches) — all sit at the same band. The
native 3 m result adds: **full radiometric quality at native resolution does not
move it either.** So the limit is not imagery compression or resolution; it is the
frozen web-pretrained features and the small labelled AOI set. Practical takeaway
for the paper: the lightweight ~7 GB JPEG subset was the right call — it cost no
accuracy.

## Pointers
- Shared report (LaTeX): `main.tex`, subsection `sec:nativeraster` — tables
  `tab:nativeraster` (caption-retrieval protocol) and `tab:native3mcv` (this
  repository's own dominant-class benchmark).
- Loader / integration: `src/datasets/dynamic_earthnet_planet.py`.
