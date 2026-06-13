# INVENTORY — what exists where (single index)

*Each machine runs a small READ-ONLY script that writes its manifest into [`inventory/`](inventory/),
commits, pushes — so every machine sees every other machine's untracked
payloads via `git pull`. Manifests are **generated, never hand-written**; trust each only as of its
scan date. Repo status → [`STATUS.md`](STATUS.md).*

## Fleet roles (policy, stable)

| Machine | Role for THIS repo | Hard rules |
|---|---|---|
| `laptop-4060` (Win 11, RTX 4060 8 GB) | **primary compute**: experiments, data, app dev | — |
| `macbook` (work) | docs, light dev, report writing | **no heavy runs** |
| `pc-1050ti` | dormant — excluded | — |
| GitHub `origin` | source of truth: code, docs, results JSONs | no secrets/identifiers, ever |
| HF Space (`space` remote) | deployed Gradio engine (the course product) | keep in sync with `main` releases |
| HF Hub (private) | cold archive for big artifacts if needed | datasets stay download-pointers |
| Colab/Kaggle | burst GPU if 8 GB VRAM short (free T4 16 GB) — allowed | — |

## Machine manifests

| manifest | last scan | notes |
|---|---|---|
| [`inventory/laptop-4060.md`](inventory/laptop-4060.md) | **regen pending (post-expansion)** | stale @ 2026-06-10 (`data/` 32.7 GB pre-expansion); current `data/` = **42 GB** (measured 2026-06-13; breakdown below) — re-run `make_inventory.ps1` to refresh the committed manifest |
| `inventory/macbook.md` | — pending | `bash ops/make_inventory.sh macbook`, commit |
| [`inventory/cloud.md`](inventory/cloud.md) | 2026-06-10 | remotes, HF Space, dataset pointers |

*`main.tex` is tracked in git (not a gitignored single copy).*

## Datasets on disk (`data/` = 42 GB, gitignored; measured 2026-06-13)

| dataset | dir | size | role |
|---|---|---|---|
| QFabric (TEOChatlas) | `data/QFabric/` | 16 GB | `qfabric_teo` + `qfabric_status` crops |
| embedding/patch caches | `data/cache/` | 11 GB | `.npz` global + patch + localization caches (regenerable; biggest reclaimable if disk gets tight) |
| Dynamic EarthNet | `data/DynamicEarthNet/` | 8.7 GB | primary; DEN npy + torchgeo meta |
| SECOND-CC | `data/_second_cc/extracted/SECOND-CC-AUG/` | 4.3 GB | captioned six-class land-cover change + per-phase semantic maps |
| LEVIR-CC **and** LEVIR-MCI | `data/_levir_mci/extracted/LEVIR-MCI-dataset/` | 2.8 GB | one shared copy — MCI is a strict superset of CC (identical pairs + captions + change masks); both `levir_cc` and `levir_mci` loaders read it |

(`.model_cache/` = 3.8 GB, separate from `data/`: the CLIP/GeoRSCLIP/RemoteCLIP weights.)

**Reclaimed this expansion (2026-06-12):** redundant download archives (LEVIR-MCI/SECOND-CC zips,
SECOND base zip/rar, `labels.tar.gz`, `Levir-CC-dataset.zip`), the dead `_torchgeo_labels` (rejected
DEN alt source), and the duplicate `_levir_cc/extracted` (deduped onto the LEVIR-MCI dir) — ~14 GB
freed. **Disk now: ~55 GB free.** If the QFabric slice needs room, `data/cache/` (11 GB,
regenerable from `scripts/export_results.py` / re-encode) is the first reclaim.

## QFabric expansion (pending access)

The remaining committed dataset is the pentatemporal + polygon-mask **QFabric** slice — full
pickup-cold spec in [`docs/QFABRIC_FUTURE_WORK.md`](docs/QFABRIC_FUTURE_WORK.md)
([`docs/DATA_EXPANSION_PLAN.md`](docs/DATA_EXPANSION_PLAN.md) Track 2/3). Source:
`labaerien/qfabric` (HF, **gated — access request awaiting Lab Aérien review**). Plan: pull a
**capped ~50-location slice** (5 dates + COCO polygon vectors, ~3–5 GB) — **never** the 298 GB
EVER-Z parquet. ~55 GB free is ample for the capped slice. No dataset imagery on the MacBook.

## How to add / refresh a machine

1. `git pull`, then from inside the repo:
   - Windows: `powershell -ExecutionPolicy Bypass -File .\ops\make_inventory.ps1 -MachineId laptop-4060`
   - macOS / Linux: `bash ops/make_inventory.sh macbook` (ops tooling lives under `ops/`)
2. Eyeball the output for anything sensitive (script sanitizes `~`, but check).
3. `git add inventory/<id>.md` → commit `inventory(<id>): refresh manifest` → push.
4. Refresh when untracked payloads change meaningfully — not every commit.
