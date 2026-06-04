# Local tasks — RTX 4060 / local data

Tasks that **must run on the local machine** (its GPU, the cached CLIP/GeoRSCLIP
weights under `.model_cache/`, and the DEN data under `data/`). They cannot run in a
sandbox without weights/data. Run **one long-running command per block**.

Currently open because the seasonal-gate work was added but its measured numbers and
the post-add test counts still need a local run (see the `_tbd_` table in `REPORT.md`
§7 and the counts in `README.md` / `REPORT.md`).

## Prerequisites (one-time)

```bash
cd <repo root>            # MSc_GBDA-OV_Temporal_Change_Retrieval
pip install -e .          # only if the current venv doesn't already have the package
```

Model weights download to `.model_cache/` automatically on first encoder use. DEN data
is expected under `data/DynamicEarthNet` (pass `--root` if yours lives elsewhere).

## 1. Fast test suite — confirm the new seasonal-gate tests pass

CPU/mock, no network. Verifies `tests/test_seasonal_gate.py` (6 new tests) is green.

```bash
pytest -q --ignore=tests/test_text_encoder.py
```

Expected: ~**203 passed** (was 197 + 6 new). The exact count feeds the doc fix below.

## 2. Seasonal-robustness gate — populate the FPR table (machine-specific)

Needs the real GeoRSCLIP encoder + DEN `.npy` data. Produces the stable-pair
Δ-similarity false-positive-rate sweep that fills `REPORT.md` §7's `_tbd_` row.

```bash
python -m scripts.run_seasonal_gate --root data/DynamicEarthNet --encoder georsclip --color-mode rgb --split test
```

- Prints: stable-pair count, mean Δ-similarity, FPR at each threshold.
- Writes: `results/seasonal_gate__georsclip__rgb__test__frac0.05.json` (`mode=recomputed`).
- Idempotent: re-running reuses the JSON unless `--force`; `--dry-run` previews the plan.
- Optional second run (GeoRSCLIP is strongest in NRG): add `--color-mode nrg`.

## 3. Real-CLIP text test (optional) — close the only locally-unverified test

Needs real CLIP weights (~45 s). Lets the headline total be a measured number rather
than the arithmetic `197 + 15 = 212`.

```bash
pytest tests/test_text_encoder.py
```

## After running — hand back two outputs

1. The `pytest` tail from **#1** → update the test counts in `README.md:245` and
   `REPORT.md` (§ lines 166 / 679 / 860).
2. The console output (or `results/…json`) from **#2** → fill the `_tbd_` FPR-table row
   in `REPORT.md` §7 (`mode=recomputed`).

(All `git add` / `commit` / `push` are run by you; this file is not committed unless you
choose to add it.)




## Two deliberate follow-ups (need your run output — I won't guess numbers)

1. **Test counts** in README.md:245 / REPORT.md:166,679,860 still say 197/212. They'll rise by 6 once the new tests run. Paste me the `pytest` tail and I'll update all of them with the verified count.
2. **The REPORT FPR table** has `_tbd_` cells. Paste the `run_seasonal_gate` output (or the JSON it writes to `results/`) and I'll fill the row (`mode=recomputed`).


A couple of notes:

* If the package isn't installed in the current venv yet: `pip install -e .` first (one-time).
* I run no git — when you're ready, the commit/push commands are the ones from my last message.

Send me the outputs from #1 and #2 and I'll finish the test counts and the FPR table in one pass.
