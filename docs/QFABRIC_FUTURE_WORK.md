# QFabric (pentatemporal + polygon masks) — future-work spec

> **DROPPED 2026-06-27 — do not pursue.** The full 5-date polygon-mask form is unobtainable: the
> `labaerien/qfabric` HF mirror stayed gated (manual review pending ~2 weeks) and **every contact
> channel is dead** — `hello@labaerien.com` bounces (GitHub-Pages domain, no mail server),
> `sagar@granular.ai` returns 550 "account inactive" (Granular folded; lead author left RS for Deltia
> AI), `engine.granular.ai` returns Cloudflare 1016 origin-DNS. No deliverable impact — the
> localization pillar is already covered by LEVIR-MCI + SECOND-CC, and Tracks 0–4 meet the full brief.
> The reduced `qfabric_teo` (2-date retrieval crops) stays in the corpus. The spec below is retained
> as a historical record only; revive solely if a working mirror with polygon vectors appears.

*Single authoritative description of the one remaining committed dataset, written so a contributor
can pick it up cold. Status as of 2026-06-13: **blocked on dataset access** (below). The current
repo already ships a *reduced* QFabric (`qfabric_teo` / `qfabric_status`, 2-date TEOChatlas crops,
no masks); this spec is the **localization upgrade** to the full 5-date polygon-mask form. Plan
context: [`DATA_EXPANSION_PLAN.md`](DATA_EXPANSION_PLAN.md) Tracks 2/3. Dataset reference card:
[`Common_Resources.md`](Common_Resources.md) §1.1.*

---

## 1. What QFabric is

QFabric (Verma et al., *Multi-Task Change Detection Dataset*, CVPR-W EarthVision 2021) — a
multi-task urban-change dataset:

- **~450,000 change polygons** across **504 locations** in 100 cities.
- **5 temporal dates** per location (pentatemporal) — true multi-date construction timelines.
- **6 change-type** classes + **9 change-status** classes (per the paper).
- High-resolution RGB; the paper works in **512×512** patches; the raw mirror tiles are larger
  (~1024², ~13 MB each — confirm on access).

Why it matters here (its **role / contribution**, distinct from the other datasets):

| dataset | gives the project | what QFabric adds beyond it |
|---|---|---|
| LEVIR-MCI | building/road change masks, **2 phases** | a **construction-domain**, **5-date** localization test (LEVIR is 2-phase) |
| SECOND-CC | six land-cover classes, **2 phases**, captions | **finer temporal pinpointing** (which of 5 dates the change happened) + construction change-type/status masks |
| current `qfabric_teo` | 6 change-type **retrieval** labels (2-date crops) | **pixel-level polygon masks** → makes QFabric a *localization* dataset, not just retrieval |

In one line: QFabric is the project's **construction-domain localization + temporal-pinpointing**
pillar. The reassessment ([`DATA_EXPANSION_PLAN.md`](DATA_EXPANSION_PLAN.md) §1) rates it ★★★ for
retrieval (regime-limited — end-state appearance dominates, zero-shot ≈ random) but **strong for
localization**, which is exactly the gap this slice fills.

## 2. Access (the current blocker)

- **Source:** [`labaerien/qfabric`](https://huggingface.co/datasets/labaerien/qfabric) on the HF Hub
  — **gated**. As of 2026-06-13 the access request is **awaiting manual review by Lab Aérien**
  (each request is reviewed by hand; released for academic / non-commercial use only, CC-BY-NC-4.0).
  Auth (HF login) is already configured on `laptop-4060`; the block is the pending approval, not the token.
- **Access-channel audit 2026-06-27 — only the HF form works; the other two channels are dead:**
  - **HF "Request access" button** (on the dataset page) is the **sole working path** — paste an
    academic/non-commercial use justification; Lab Aérien reviews by hand. Request text drafted in
    the session scratchpad (`qfabric_access_drafts.md`).
  - `hello@labaerien.com` (listed contact) is for **commercial / bulk transfer only** *and* it
    **hard-bounces** — `labaerien.com` resolves to GitHub Pages (185.199.108–111.153), which runs
    no mail server (SMTP connect times out). Do not rely on it for access.
  - `engine.granular.ai` (the Granular project link below) is **offline** — Cloudflare **error
    1016, origin DNS unresolvable**. The original self-serve platform is gone.
  - **`sagar@granular.ai` is also DEAD** (550 "account inactive / User unknown", 2026-06-27) —
    Granular has folded and lead author Sagar Verma has left RS (now at Deltia AI). So **every email
    channel is dead**; the HF form is the only live automated path. Remaining live human contacts:
    Sagar via LinkedIn `in/versag` / X `@ver_sag` (low odds — off RS) or co-author Aakaash Panigrahi
    (`aakaash-panigrahi.com`). Drafts: `data/_notes/qfabric_access_drafts.md` (untracked).
  - Review pending ~2 weeks with a dead contact mailbox → **Lab Aérien may be dormant; approval is
    not guaranteed.** **Realistic outcome: accept QFabric stays cut** — Tracks 0–4 already meet the
    brief; this slice is a localization bonus, not a gate.
- **Do NOT use** [`EVER-Z/QFabric_mt_images_1024`](https://huggingface.co/datasets/EVER-Z/QFabric_mt_images_1024)
  — it is **298 GB**, image-only (no polygon vectors), and out of scope by size. The `labaerien`
  mirror is strictly better: it carries both the 5-date rasters **and** the COCO polygon vectors.
- Other references: Granular AI engine (**offline, CF 1016**) — https://engine.granular.ai/organizations/granular/projects/631e0974b59aa3b615b0d29a/overview ;
  paper — https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/papers/Verma_QFabric_Multi-Task_Change_Detection_Dataset_CVPRW_2021_paper.pdf

**Automated watcher (active):** a session cron job (id `201d9dbc`, every 2 h at :17) probes a gated
file; the moment it returns instead of `403 GatedRepoError`, it deletes itself and auto-starts the
build below. The probe:
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('labaerien/qfabric','vectors/random-split-1_2023_02_05-11_47_30/COCO/63df8f4e08b5b0034d8044eb.json',repo_type='dataset'); print('GRANTED')"
```
Caveat: the watcher is **session-only** (it lives only in the local session that scheduled it) and
auto-expires in 7 days. If it lapses, just re-run the probe manually once the HF approval email lands.

## 3. On-disk structure of the `labaerien/qfabric` mirror

Confirmed from the repo file listing (metadata is readable even while file content is gated):

```
labaerien/qfabric/
├── rasters/raw/<loc>.d<N>.<MMDDYYYY>.tif     # 503 locations × 5 dates (d1..d5); ~13 MB/tile
│     e.g.  0.d1.12202015.tif … 0.d5.12032018.tif
├── vectors/random-split-1_<ts>/COCO/<id>.json   # 6,083 COCO-format polygon annotation files
├── assets/  cache/  README.md
```

- **2,485 raster files** = 503 locations × ~5 dates. Filename = `<location_id>.d<date_index>.<date>.tif`.
- **6,083 vector JSONs** in COCO format (polygons + `category_id`); the change-type and change-status
  taxonomies map onto COCO categories.
- **CONFIRM ON ACCESS (could not be read while gated):** (i) the exact COCO `categories` schema —
  which `category_id` ↔ which of the 6 change-types / 9 status classes; (ii) raster dimensions,
  band order, and dtype; (iii) how a vector JSON keys to its raster(s) (per-location? per-date?).
  Verify these the same way the LEVIR-MCI mask palette and SECOND-CC palette were verified before
  trusting them (extract uniques / read one sample).

## 4. Disk budget — capped slice

Per-location footprint ≈ 5 dates × ~13 MB ≈ **65 MB**. A **~50-location** capped slice is therefore
**~3.25 GB rasters + ~0.5–1 GB vectors ≈ 3–5 GB** total. (An earlier plan estimate of "10–15 GB"
was a loose upper bound for a larger slice; the computed figure for 50 locations is ~3–5 GB. Scale
linearly: ~100 locations ≈ 6–10 GB.) Disk on `laptop-4060` is currently **~55 GB free** — ample.
**Cap rule:** download a fixed location subset via `hf_hub_download` `--include` glob patterns;
record the cap (locations × dates) in the loader docstring. Never pull the whole mirror unfiltered
and never the 298 GB EVER-Z parquet.

## 5. Build steps (the approved approach)

1. **Verify schema** (§3 confirm-on-access items) — read one COCO json + one raster header.
2. **Pull the capped slice** — `hf_hub_download` (or `snapshot_download` with `allow_patterns`) for
   ~50 chosen `<loc>.d*.tif` + their vector JSONs into `data/_qfabric_mt/` (gitignored under `data/`).
3. **Loader** — `src/datasets/qfabric_localization.py` (new file; extends/sits beside
   `qfabric_teo`, does **not** edit it). Serve the 5-date rasters as bi-temporal pairs (pairing
   choices: d1→d5 for max change, or adjacent dN→dN+1 for temporal pinpointing — expose a
   `pairing` option), and **rasterize the COCO polygons to per-change-type / per-status masks**
   (polygon fill at raster resolution) exposed via `load_change_mask(pair, change_class)` +
   `has_mask` — the **same interface** `levir_mci` and `second_cc` already implement, so it drops
   straight into `eval_localization.py`. Reuse `qfabric_teo`'s `max_per_class` capping.
3a. **Register** in `src/datasets/registry.py` (factory + the shared `_levir_cc_opts`-style adapter)
    and add `src/queries/qfabric_localization.py` (queries over the 6 change-types + key statuses,
    with a `QUERY_TO_MASK_CLASS` map mirroring `second_cc`/`levir_mci`). Import in
    `src/queries/__init__.py`.
4. **Tests** — `tests/test_qfabric_localization.py` with a tiny synthetic fixture (a couple of
   rasters + a COCO json + rasterized masks), mirroring `tests/test_second_cc.py`: lists pairs,
   change-type labels, `load_change_mask`/mask values, registry build, queries registered.
5. **Benchmark + localization** — `scripts/benchmark_qfabric_localization.py` (retrieval per-query
   AP, mirror `benchmark_second_cc.py`) **and** `python -m scripts.eval_localization --dataset
   qfabric_localization` (the generalized script already dispatches on `--dataset` + the dataset's
   `QUERY_TO_MASK_CLASS`; just add the dataset to its `_DATASETS` map). Add progress bars to encode
   loops. Cache patch embeddings under `data/cache/` (gitignored).
6. **Write up** — REPORT new §7.14 (retrieval + localization, honest framing vs the random-patch
   floor, same as §7.12/7.13), main.tex subsection, STATUS §2/§3, and mark
   [`DATA_EXPANSION_PLAN.md`](DATA_EXPANSION_PLAN.md) Tracks 2/3 QFabric DONE. Then hand over
   commit + push commands (user runs all git).

## 6. Guardrails (do not violate)

- **Never edit shared pipeline files** (`embeddings.py`, `retrieval.py`, `benchmark.py`, `train.py`,
  `app.py`, `scripts/run_pipeline.py`) — add new files via the registry, like every other dataset.
- Frozen backbones, zero-shot + PEFT-light only; stay inside the Case-11 brief.
- Honest reporting: QFabric retrieval is expected weak (regime-limited); the *localization* result
  is the contribution. Report against the random-patch floor; flag classes with too few positives.
- Expect localization to be **weak** (LEVIR-MCI and SECOND-CC both showed the change heatmap is a
  weak localizer, lifts within ±0.04–0.10) — QFabric tests whether construction polygons + 5 dates
  change that verdict. Report the honest answer either way.
