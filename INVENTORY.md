# INVENTORY — what exists where (single index)

*Each machine runs a small READ-ONLY script that writes its manifest into [`inventory/`](inventory/),
commits, pushes — so every machine sees every other machine's untracked
payloads via `git pull`. Manifests are **generated, never hand-written**; trust each only as of its
scan date. Repo status → [`STATUS.md`](STATUS.md).*

## Fleet roles (policy, stable)

| Machine | Role for THIS repo | Hard rules |
|---|---|---|
| `laptop-4060` (Win 11, RTX 4060 8 GB) | **primary compute**: experiments, data, app dev | — |
| `aris` (HPC) | **FORBIDDEN for course work** | ARIS access = thesis-only; the GBDA clone there is a thesis *library checkout* (pinned v1.0), never a course workbench |
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

No `aris` manifest by design — see fleet rules.

## How to add / refresh a machine

1. `git pull`, then from inside the repo:
   - Windows: `powershell -ExecutionPolicy Bypass -File .\ops\make_inventory.ps1 -MachineId laptop-4060`
   - macOS / Linux: `bash ops/make_inventory.sh macbook` (ops tooling lives under `ops/`)
2. Eyeball the output for anything sensitive (script sanitizes `~`, but check).
3. `git add inventory/<id>.md` → commit `inventory(<id>): refresh manifest` → push.
4. Refresh when untracked payloads change meaningfully — not every commit.
