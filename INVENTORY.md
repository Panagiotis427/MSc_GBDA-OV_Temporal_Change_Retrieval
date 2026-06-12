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
| [`inventory/laptop-4060.md`](inventory/laptop-4060.md) | 2026-06-10 | `data/` 32.7 GB / 271,860 files; `.model_cache/` 3.8 GB (RemoteCLIP etc.); **`main.tex` = gitignored single copy** |
| `inventory/macbook.md` | — pending | `bash ops/make_inventory.sh macbook`, commit |
| [`inventory/cloud.md`](inventory/cloud.md) | 2026-06-10 | remotes, HF Space, dataset pointers |

## Planned-expansion disk constraint (2026-06-12)

The data-expansion plan ([`docs/DATA_EXPANSION_PLAN.md`](docs/DATA_EXPANSION_PLAN.md) §3) is
**disk-gated**: `laptop-4060` reports **57 GB free** with **32.7 GB already in `data/`**, and the
committed datasets add ~20–27 GB (LEVIR-MCI 2.77 GB · SECOND-CC ~5 GB · QFabric localization slice
~10–15 GB capped · embedding caches ~2–4 GB) — projecting to **~53–60 GB, at/over the ceiling**.
**Prerequisite:** run a per-dataset `du -sh data/*` on the 4060 (manifests are dir-level only, so
the current 32.7 GB breakdown is unknown) and reclaim stale caches/duplicates before downloading.
Never pull the 298 GB EVER-Z QFabric parquet; cap the QFabric slice. No dataset imagery on the
MacBook.

## How to add / refresh a machine

1. `git pull`, then from inside the repo:
   - Windows: `powershell -ExecutionPolicy Bypass -File .\ops\make_inventory.ps1 -MachineId laptop-4060`
   - macOS / Linux: `bash ops/make_inventory.sh macbook` (ops tooling lives under `ops/`)
2. Eyeball the output for anything sensitive (script sanitizes `~`, but check).
3. `git add inventory/<id>.md` → commit `inventory(<id>): refresh manifest` → push.
4. Refresh when untracked payloads change meaningfully — not every commit.
