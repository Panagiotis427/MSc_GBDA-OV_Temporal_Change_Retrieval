# Laptop Runbook — finalize Case-11 (run on the GPU machine)

The three gates that can't run on the no-GPU machine, with the exact REPORT edits each one feeds.
**Prereqs:** `pip install -e .`; DEN + QFabric data present; QFabric crops already extracted
(the one-time `benchmark_qfabric --extract-from ... --extract-only` step). Commit `c7e7017` is the baseline.

---

## Gate 1 — QFabric re-run  ⚠️ REQUIRED (today's label-join fix changes labels)

```bash
# 1. Rebuild labels with the fixed matcher. Note the new distribution + conflict count.
python -m scripts.build_qfabric_labels --out data/QFabric/qfabric_teo_labels.json

# 2. §7.8 — naive/zero_shot on the stratified eval subset (re-encodes: subset membership changes with labels)
python -m scripts.benchmark_qfabric --crops-root data/QFabric/teochat_crops \
    --max-per-class 120 --results-dir results

# 3. §7.9 — PEFT on the held-out train/test split
python -m scripts.benchmark_qfabric --crops-root data/QFabric/teochat_crops \
    --max-per-class 120 --results-dir results --peft
```

Then update `REPORT.md` from the **new** `results/qfabric_teo__*.json`:
- **§7.8 table** (CLIP/GeoRSCLIP/RemoteCLIP × naive/zero_shot mAP).
- **§7.8 prose** — *re-derive ALL of these from the new JSON*, they were edited against the OLD labels:
  the macro-prevalence baseline (was 0.167), the per-query APs (road/residential/industrial/…),
  and the `mega_projects (… pairs)` count (the "76" I set was the old-label value).
- **§7.9 table** (train/test × naive/zero_shot/peft).
- Regenerate the figures (below), then re-read the §7.8/§7.9 *conclusions* — the naive-vs-zero_shot
  and PEFT-helps-on-QFabric narratives may change, not just the digits.

## Gate 2 — fast suite + true test count

```bash
pytest -q --ignore=tests/test_text_encoder.py
```
- Confirm the new tests pass: `test_dynamic_earthnet_pp.py` (4), `test_metrics.py`, `test_cache_tag.py`,
  `test_train_loss.py`. If `test_dynamic_earthnet_pp.py::test_compose_ndvi_values` fails on the `191`
  cell, it's a numpy-version float nuance, not a real bug — tell me the actual value.
- Replace the stale **`129`** with the printed pass count in three places:
  `REPORT.md` §5 (~L157), `REPORT.md` §8 timings (~L599), `QUICKSTART.md` (~L142).

## Gate 3 — back §7.5 re-ranking table (no committed artifact today)

§7.5's diversity/coherence numbers (`src/rerank.py` exists, but no eval script writes a JSON).
Pick one:
- **(a)** Have me draft `scripts/eval_rerank.py` (apply `src.rerank.Reranker` to the GeoRSCLIP+NRG
  zero_shot test ranking, recompute mAP/Recall@K, write a JSON) → then the table is artifact-backed.
- **(b)** Caveat §7.5 as an interactive/UI measurement (cheapest; honest).

## Figures (after Gate 1)

```bash
python -m scripts.export_results --color-modes rgb nrg ndvi \
    --approaches naive zero_shot peft --lora --confusion --results-dir results
python -m scripts.make_figures --results-dir results --out-dir assets/figures
```
Verify the QFabric plots updated (`assets/figures/qfabric_peft_test.png`, `map_bars__eval__rgb.png`).
If `make_figures` doesn't emit the QFabric-specific plots, check how they were originally produced.

## Then commit + push the finalized numbers

```bash
git add -p && git commit -m "QFabric §7.8/§7.9 re-run after label fix; true test count; §7.5 backing"
git push
```

> Ping me with the new label distribution + the pytest count and I'll help reconcile the REPORT cells.
