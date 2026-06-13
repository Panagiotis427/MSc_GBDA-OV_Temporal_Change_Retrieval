# UX / UI design — toward a sharp public app

*Living design doc for the Gradio "semantic change search engine" (`src/app.py`,
deployed as a HuggingFace Space). Goal: a sharp, public-facing demo.*

> **Build/verify constraint.** Visual/interaction work must be **seen rendered** to be
> done well, and any data-bearing feature needs the encoder weights + encoding compute.
> Items below are split into **done offline** (verifiable by the test suite without a
> render) vs **needs a render/GPU session**. Do not blind-ship layout or retrieval
> changes to a public Space — verify them live first.

---

## Done (offline, this round)
- **Friendly dropdown labels** — Dataset/Encoder dropdowns show proper names ("Dynamic
  EarthNet", "QFabric — change type", "CLIP ViT-L/14", …); the value sent to callbacks
  stays the registry key. (`DATASET_LABELS`/`ENCODER_LABELS` + `_labeled`.)
- **Honest score framing** — the 0–1 number is a **relative match score** (min–max
  normalized over the candidate set), not a calibrated probability. Pill/table/caption
  relabelled "confidence" → "match"; pill carries a clarifying tooltip.
- **Sharper first-query progress message** — the existing `gr.Progress` now flags that
  the first query on a dataset/approach encodes the corpus (a few seconds).

## Done (offline, 2026-06-13 round) — build-verified (full Gradio tree constructs; app tests green)
- **Dataset labels completed** — `DATASET_LABELS` was missing the datasets added this expansion;
  `second_cc` and `levir_mci` showed raw registry keys in the dropdown. Added friendly,
  role-bearing labels for all six corpora (e.g. "SECOND-CC — 6-class land cover (+ masks)").
- **Curated examples that actually retrieve** — the example fill-buttons were DEN-generic and
  included **zero-positive** queries (seasonal snow = 0 positives → always noise) and at-chance
  ones. Replaced with DEN's honest signals (wetland-formation, the only robustly above-chance
  signal, + construction which patch scoring recovers); default query set to a wetland one so the
  **first impression actually works**. Caption states the honesty up front.
- **Honest-expectation note** — header + examples caption + About now state this is a research demo
  and retrieval is approximate (≈0.20 mAP ceiling). (Pairs with the match-score reframing.)
- **"About / How it works" disclosure (Idea 1, partial)** — a collapsed accordion at the top
  consolidates: what it is, honest expectations, the four approaches, how to read the match score +
  heatmap, and the corpus selector note. Default view stays clean (collapsed); the verbose
  explanation no longer clutters the query area.
- **All processed corpora are switchable in-app now (functional fix, was broken).** The Dataset
  dropdown listed corpora but `reload()` reused the DEN root, so picking anything non-DEN **errored
  on select**. Added `DATASET_PROFILES` (per-corpus root / split / colour / loader-extras) resolved
  in `reload()`, with a `loader_extra` dict on `RunConfig` threaded into `build_dataset` so QFabric's
  `labels_path` / `max_per_class` reach its loader. Verified all six load + return query results
  in-app: LEVIR-CC (1929), LEVIR-MCI (1929), SECOND-CC (1227), QFabric-type (2476), DEN (110),
  QFabric-status (4200). The dropdown is **sorted best-result-first** (`DATASET_RANK`): LEVIR-CC →
  LEVIR-MCI → SECOND-CC → QFabric-type → DEN → QFabric-status. DEN stays the pre-selected default
  (its curated examples are tuned to it). Profile roots are **Space-safe**: `reload()` falls back to
  the current root when a profile's data dir is absent (e.g. the fixture-only HF Space), so a
  non-fixture pick there errors gracefully instead of crashing. QFabric first query re-encodes its
  crops (no matching app-tag cache) — a few seconds, progress shown.

## Idea 1 — progressive disclosure (default view stays clean) · needs render
Default view = **query box → ranked results**, nothing else. Everything verbose moves
behind disclosure:
- Long explanations → an **"About / How it works"** expander (collapsed) + `?` tooltips
  (`info=`) per control.
- Settings / Filters already collapse — keep.
- Per-result detail (raw score, permanence rationale, methodology caveats) → a per-result
  expander rather than always-on text.
- *Risk:* layout/spacing must be checked rendered; the move itself is low-risk Gradio
  structure (Accordion / `info=`).

## Idea 2 — search across all (available) datasets at once · needs render + data
Compelling demo of the dataset-agnostic engine: one query → ranked change events from
across datasets, each tagged with its **source dataset + timestep**. Two hard problems and
their fixes:
1. **Public-Space feasibility** (can't hold raw DEN ~7 GB + QFabric + LEVIR-CC). **Fix:**
   precompute **pair embeddings** (small `.npz`) + thumbnails *offline*; ship/host those.
   Runtime only encodes the **query text** and cosine-ranks the merged store — no raw data
   or GPU at Space runtime. **Bonus: also removes the slow first-query encode → instant search.**
2. **Cross-dataset scores are not comparable** (regimes/scales differ — §7.8 naive-wins vs
   §7.10 Δ-wins; raw magnitudes differ). A naive global ranking is dominated by one
   dataset's scale → junk. **Fix:** rank-based or per-dataset z-normalized **fusion** before
   merging. Mandatory.
- **Default = all** is reasonable **only if availability-driven** (search whatever
  embeddings are present; never assume all datasets exist), with explicit dataset selection
  to narrow. **Source/timestep badges on every result** — worth doing even single-dataset.
- v1 scope (correct + simple): one encoder; per-dataset sensible color; per-dataset score
  normalization; rank-fuse; tag source. Per-dataset "best config" tuning is a later refinement.

## Other ideas (priority-ordered) · render unless noted
- **Curated "try these" examples that actually retrieve well** — the engine is honestly
  weak (~0.20 ceiling), so hand-pick queries that *work* (wetland-formation→DEN,
  road/residential→QFabric, new-buildings→LEVIR-CC) so first impressions aren't noise. *High impact.*
- **Honest expectation note** — a subtle "research demo — retrieval is approximate" line
  (pairs with the match-score reframing already done). *Integrity + UX; partly offline.*
- **Before/after swipe slider** on T1/T2 — the classic, compelling RS-change interaction.
- **Heatmap opacity/toggle** + crisp T1/T2 labels.
- **Mobile-responsive layout** — public visitors arrive on phones.
- **Precomputed-embeddings → instant, cheap Space** (the Idea-2 unlock; also a standalone win).
- **Shareable result permalinks**.

## Suggested phasing
1. *(done)* labels, honest score, progress message.
2. **Precompute-embeddings backend** — unlocks instant search **and** Idea 2; biggest leverage.
3. **Progressive disclosure** layout pass (render session).
4. **Search-all + source badges + fusion** on the precomputed store.
5. Polish: curated examples, swipe slider, heatmap toggle, mobile, permalinks.
