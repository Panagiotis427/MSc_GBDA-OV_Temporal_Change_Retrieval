# Data Expansion & Honest-Reframe Plan

*Created 2026-06-12. **Planning document — pre-implementation.** Captures the dataset
reassessment, the disk budget, and the work tracks agreed before any edits or downloads.
Single source of repo status remains [`STATUS.md`](../STATUS.md); canonical results remain
[`REPORT.md`](../REPORT.md). This plan is cited from `STATUS.md §4`. When a track lands, its
outcome moves into `STATUS.md` / `REPORT.md` and this doc is updated, not duplicated.*

---

## 0. Why this plan exists

The deliverable is functionally complete, but the retrieval numbers are weak and were, in
places, **framed around non-representative sub-cases** (notably the DEN GeoRSCLIP+NRG
zero-shot `0.426`, which is a single lucky high-wetland fold — REPORT Appendix B.8). This plan
does two things, in scope of the Case-11 brief
([`GBDA_Case11_Overview.md`](GBDA_Case11_Overview.md)):

1. **Reframe** the results around genuine, general, cross-validated numbers — never single
   folds/queries/splits — across all datasets, tasks, modes, and models.
2. **Expand the data toward purpose-fit datasets** (open-vocabulary change with real change
   labels and localization masks), after verifying *why* the current numbers are low.

### Root-cause verdict (verified, not assumed)

The low numbers are **not** primarily a "small subset" problem. The reassessment (sources in
§1) found a per-dataset split:

- **DEN** — the **full 75-AOI dataset** was used (the entire dataset; 825 pairs), with **dense,
  complete per-pixel monthly labels**. The weakness is (a) an **intrinsic change-type skew**
  (built for segmentation: snow = 0 positives, buildings/urban/deforestation rare,
  ~44 of 71 dominant-flips are wetland↔agriculture) and (b) a **method ceiling** (global VLM
  embedding-differencing dilutes localized change; patch scoring lifts it to ~0.19 but no
  further). No alternative DEN source fixes either — all sources carry the same labels.
- **QFabric** — a **heavily reduced form** was used (TEOChatlas before/after crops, stratified
  N≈2,476), discarding 3 of QFabric's 5 temporal dates **and** all polygon masks. A richer
  source exists (pentatemporal + change-type/status masks). More data tightens CIs but does not
  change the regime finding (end-state appearance > temporal Δ; zero-shot ≈ random).
- **LEVIR-CC** — the **success case** (macro mAP ~0.50–0.57), but **under-utilized**: only the
  test split + 3 queries were scored, and the loader already parses **vegetation + water tags
  that have no queries**.

So: the honest story is **purpose-mismatch + method ceiling**, plus genuine head-room in the
change-captioning family. This plan acts on the head-room and retires the misleading framing.

---

## 1. Dataset reassessment (deep scan, 2026-06-12)

Relevance is judged for **open-vocabulary temporal change *retrieval*** specifically.

| candidate | purpose-fit | disk | verdict |
|---|---|---|---|
| **LEVIR-MCI** ([`lcybuaa/LEVIR-MCI`](https://huggingface.co/datasets/lcybuaa/LEVIR-MCI)) | ★★★★★ — strict superset of the success case | **2.77 GB** | **COMMIT.** Same 10,077 pairs + 5 captions each **+ change-detection masks for buildings & roads** (40k+ masks). Adds pixel-level localization labels with the lowest possible friction. |
| **SECOND-CC** ([arXiv 2501.10075](https://arxiv.org/html/2501.10075v1); base **SECOND**, 4,662 pairs / [captain-whu SCD](http://www.captain-whu.com/project/SCD)) | ★★★★★ — 6 land-cover classes → **30 change categories** | ~5 GB | **COMMIT.** 6,041 pairs / 30,205 captions, built to be harder/more diverse than LEVIR-CC. The genuine open-vocabulary **breadth** test the current datasets lack; SECOND base carries semantic change masks for localization. |
| **QFabric (full / pentatemporal)** ([Granular AI](https://www.granular.ai/resources/blog/qfabric:-multi-task-change-detection-dataset); avoid [EVER-Z 298 GB](https://huggingface.co/datasets/EVER-Z/QFabric_mt_images_1024)) | ★★★ — regime-limited for retrieval; strong for localization | ~10–15 GB (capped) | **COMMIT (disk-gated).** 5 dates + polygon change-type/status masks. Buys **localization + time-step pinpointing**, not retrieval mAP. Use a capped slice — **never** the 298 GB EVER-Z parquet. |
| **DEN finer temporal** (existing frames) | ★★★ — time-step pinpointing the brief values | ~0 | **COMMIT.** No new imagery: add monthly (24-step) pairing alongside the current bimonthly to sharpen temporal localization. |
| **DEN alternative sources** (TUM 525 GB raw, HEVC video, torchgeo) | — | — | **REJECT.** Same 75 AOIs, same monthly 7-class labels. No source adds change-type diversity or richer labels; source swap ≠ better data. |
| **fMoW** ([arXiv 1711.07846](https://arxiv.org/abs/1711.07846)) | ★ — **no change labels** | — | **REJECT, with documented reasons (see §2).** |

**True-to-purpose numbers to quote going forward** (macro over all evaluable queries,
cross-validated where applicable, vs the random-ranking floor — never a single fold/query):

| dataset (fairest eval) | random floor | honest macro number | reading |
|---|---|---|---|
| DEN (75-AOI, 5-fold CV, fraction relevance, 9 queries) | ~0.083 | global zero-shot **0.139 ± 0.024**; best patch_top3 **0.193 ± 0.051**; only **4/9** above random | weak |
| QFabric change-type (N≈2,476, 6 queries) | 0.167 | naive **0.27**; zero-shot **0.18 ≈ random** | weak |
| QFabric status (6 queries) | 0.045 | **0.057–0.084** | at floor |
| LEVIR-CC (test, 3 queries) | 0.342 | **0.50–0.57** | strong |

The general claim: *open-vocabulary change retrieval with frozen CLIP-variant encoders recovers
change in proportion to its visual salience — strong on large, high-contrast change
(LEVIR ~0.55), at or barely above random on subtle land-cover change (DEN ~0.15) and
construction-type retrieval (QFabric ~0.27).* The `0.426` is retired from all headlines.

---

## 2. fMoW — rejection rationale (kept for the brief's defense)

The brief names fMoW, so the deviation is documented rather than silent. fMoW is a **functional
classification** dataset (63 categories such as *airport*, *stadium*); its temporal sequences
are multiple views to aid classifying a location, and the **label is the functional category,
which is static across the sequence**. It ships with **no change annotations**. Repurposing it
for change retrieval requires manufacturing pseudo-change pairs, which reintroduces the exact
**weak-label noise** that already sank DEN PEFT (a "no-change" assumption on pairs that may have
genuinely changed). fMoW is therefore the **least purpose-fit** candidate and is **not
implemented**; the project instead uses the change-captioning family (LEVIR-MCI, SECOND-CC),
which is built for the open-vocabulary change task. (Supersedes the speculative
`fMoW-Sentinel pipeline` item in [`EXTENSIONS.md`](EXTENSIONS.md).)

---

## 3. Disk budget (all work runs on the RTX 4060 — never the MacBook)

The 4060 reports **57 GB free** on the repo drive with **32.7 GB already in `data/`**
([`inventory/laptop-4060.md`](../inventory/laptop-4060.md)). The MacBook holds **no dataset
imagery and must not** — it is docs/light-dev only.

| item | est. disk | notes |
|---|---|---|
| current `data/` (DEN npy + QFabric crops + LEVIR-CC) | 32.7 GB | per-dataset breakdown **unknown** — `du` first (manifests are dir-level only) |
| LEVIR-MCI | 2.77 GB | confirmed |
| SECOND-CC | ~5 GB | estimate |
| QFabric localization slice | ~10–15 GB | **capped**; never the 298 GB EVER-Z |
| DEN monthly pairing | ~0 | reuses existing frames |
| new embedding caches (all) | ~2–4 GB | `.npz`, keyed by split + color mode |
| **projected total** | **~53–60 GB** | **at/over the 57 GB ceiling** |

**Prerequisite + mitigations (Track 0, before any download):**
1. On the 4060, run a per-dataset `du -sh data/*` and `.model_cache` audit; reclaim stale caches
   / byte-identical duplicates (to `trash/` per the never-delete rule) — the 32.7 GB likely
   contains reclaimable intermediate artifacts.
2. **Hard-cap** the QFabric slice; record the cap (locations × dates) in its loader docstring.
3. If still tight: external drive, or Colab/Kaggle burst (policy-allowed) for QFabric extraction
   only, keeping the repo-side artifact small.

---

## 4. Work tracks

All tracks: **frozen backbones**, zero-shot + PEFT-light only, RTX 4060 (or burst), and
**datasets/encoders are added as new files via the registry — shared pipeline files are never
edited** (`embeddings.py`, `retrieval.py`, `benchmark.py`, `train.py`, `app.py`,
`scripts/run_pipeline.py`). See [`ARCHITECTURE.md`](ARCHITECTURE.md) / [`EXTENSIONS.md`](EXTENSIONS.md).

### Track 0 — Disk audit (prerequisite)
- `du` per-dataset on the 4060; reclaim; confirm head-room for the budget in §3.
- **Verify:** free space ≥ projected total + 10% margin before any download.

### Track 1 — Honest reframe (no compute)
- Retire `0.426` and every single-fold/single-query/single-mode "good case" from **headlines**
  in `REPORT.md`, `main.tex`, `README.md`, and stray quotes (e.g. `EXTENSIONS.md:12`).
- Lead with the §1 true-to-purpose numbers and the salience claim.
- Keep the `0.426 → 0.10` collapse as a **featured rigor example** in REPORT Appendix B
  (random baseline + permutation p + BH-FDR + 5-fold CV) — a strength, not a hidden footnote.
- **Verify:** grep the repo for `0.426` and confirm every surviving instance is an explicit
  cautionary/appendix context, never a headline.

### Track 2 — Data integration (new loaders + query sets)
- **LEVIR-MCI:** extend the existing `levir_cc` path — add masks; use **all splits**; add the
  **vegetation + water queries** the loader already supports but never exercised.
- **SECOND-CC:** new `src/datasets/second_cc.py` (`TemporalDataset`) + `src/queries/second_cc.py`
  (queries spanning its 30 change categories); register.
- **QFabric localization:** new loader serving the capped pentatemporal slice + polygon masks
  (extends, does not replace, `qfabric_teo`); reuse `max_per_class` capping.
- **DEN finer temporal:** add a monthly pairing run (loader already supports `monthly`); no new
  data.
- **Verify:** `pytest` fast suite green; each new dataset benchmarks end-to-end with **no edits**
  to shared pipeline files; embeddings cached and reused on rerun.

### Track 3 — Localization / heatmap evaluation (PILLAR)
- Use the new mask labels (LEVIR-MCI, SECOND, QFabric polygons) to make the heatmap deliverable
  **quantitative**: pointing-game accuracy and/or mask-IoU of the patch-similarity heatmap vs the
  ground-truth change mask. This directly completes the brief's required *"heatmap highlighting
  the specific spatial region of the change"*, currently only qualitative.
- New `src/localization_eval.py` (or `scripts/eval_localization.py`); never edits shared files.
- **Verify:** localization metric reproduces from cache; reported per-dataset with the random
  floor; honest about which datasets lack masks (DEN: no instance masks → patch-IoU only).

### Track 4 — Bounded method check (optional, low priority)
- One capped learned change-attention + augmentation run to confirm/deny the memorization prior
  (REPORT B.5/B.12). Reported as an honest positive **or** negative — not a centerpiece.
- **Verify:** cross-validated, leakage-free; compared against frozen zero-shot within fold
  variance.

---

## 5. Scope guardrails (do not overshoot the brief)

- Stay inside Case-11: frozen CLIP/GeoRSCLIP/RemoteCLIP, zero-shot + PEFT-light, Recall@K/mAP,
  seasonal-drift error analysis, Gradio engine. No new training beyond PEFT-light. No new
  encoders unless an existing extension point already covers it.
- S1 SAR fusion stays **future-work** (off-brief, modality mismatch — `EXTENSIONS.md` SAR section)
  unless explicitly pulled in.
- Course work stays self-contained; the thesis pins this repo at a release and must not be
  perturbed by this plan.

---

## 6. Open decisions / risks

- **Disk (highest risk):** QFabric localization is disk-gated on Track 0's outcome. If the 4060
  cannot clear head-room, QFabric drops to burst-only extraction or future-work.
- **QFabric source:** the richer pentatemporal + masks needs the Granular login (or a documented
  capped extraction); the EVER-Z 298 GB parquet is out of scope by size.
- **Reframe aggressiveness:** plan assumes full retire of `0.426` from headlines (recommended);
  confirm before Track 1 edits land.
- **SECOND-CC / LEVIR-MCI are beyond the three named datasets** — justified by the open-vocabulary
  goal and the precedent that LEVIR-CC (also an addition) is the best result; recorded here for
  defense.

---

## 7. Sequencing

Track 0 → Track 1 (no compute, can run in parallel) → Track 2 (per-dataset, pipeline-style) →
Track 3 (depends on Track 2 masks) → Track 4 (optional, last). STATUS.md and REPORT.md are
updated in the same commit as each landing.
