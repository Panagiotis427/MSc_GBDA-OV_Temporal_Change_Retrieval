# Repository Audit — Open-Vocabulary Temporal Change Retrieval (GBDA Case 11)

**Date:** 2026-06-02 · **Commit audited:** `c6d3c22` · **Branch:** `main`

**Method:** 11-dimension adversarial audit. Each dimension was reviewed independently; every
material finding was then re-checked by a separate skeptic that attempted to refute it.
**52 of 53 verified findings were confirmed, 1 refuted.** All claim-vs-evidence checks were
run against the *committed* artifacts (`results/*.json`, `macro_summary.csv`, `assets/figures/`)
— no datasets or model weights were downloaded.

## Verdict

The **core is sound**: the three scoring formulas (`naive` / `zero_shot` / `peft`) match the
contract and all route through one `ChangeRetriever.score_all`; mAP / Recall@K / AP math is
correct; architecture invariants hold (extension-points only, no shared-pipeline edits for
QFabric); the cache is correctly keyed by split+colour; no secrets; all doc links resolve; and
nearly every reported number traces to a committed artifact.

But **"essentially complete" was over-stated.** There is one correctness bug that taints
reported QFabric numbers, several claim/disclosure issues, real test-coverage gaps on critical
paths, and the repo is **not yet usable as the thesis Q2/Q4 component** (4 high-severity
integration gaps). Standalone Case-11 grade: close, after the fixes below. Thesis-reuse grade:
not ready.

**Status legend:** ✅ fixed this session · 🖥️ needs the other laptop (data/GPU/env) ·
🤔 deferred — your decision (structural/behavioural) · 📝 small doc fix queued for next batch ·
🔒 lives in a protected shared-pipeline file (fixed doc-side only).

---

## ✅ Fixed this session (local, no data/GPU — review the diff and commit)

| Finding | File | Change |
|---|---|---|
| §1 headline CLIP test mAP disagreed with §7.3 + CSV | [REPORT.md](REPORT.md) §1 | NRG `0.102→0.104`, NDVI `0.062→0.064` |
| §7.8 mega_projects pair count wrong vs JSON | [REPORT.md](REPORT.md) §7.8 | `80 pairs → 76 pairs` |
| §7.9 "0.428→0.039" traces to no committed cell | [REPORT.md](REPORT.md) §7.9 | → `GeoRSCLIP RGB: zero-shot 0.299 → PEFT 0.041` (real §7.2 cells) |
| §9 cache-key note understated the real key | [REPORT.md](REPORT.md) §9 | now `(dataset, encoder, split, colour mode)` + `_lora` tag |
| Duplicate `pip install -e .` (no-dup-installs rule) | [REPORT.md](REPORT.md) §8 | §8 prose reworded; canonical command stays in §9 |
| Headline 0.426 basis + cross-split query-subset undisclosed | [REPORT.md](REPORT.md) §7.2 | added "Evaluation basis" footnote (3 wetland queries / 15 positives; zero-positive queries dropped) |
| Individual-stage commands inconsistent / train-on-test footgun | [QUICKSTART.md](QUICKSTART.md) | added run_pipeline-is-canonical note; `src.train` now shows `--split train` |
| `embeddings.py` docstring showed old untagged cache name | [src/embeddings.py](src/embeddings.py) | docstring now shows tagged path |
| `lora_train` default `--dataset dynamic_earthnet_pp` is unregistered → crashes | [src/lora_train.py](src/lora_train.py) | default → `dynamic_earthnet` (matches `run_pipeline`) |
| **QFabric label join matched class names as substrings** (`crossroad`→`road`) and broke multi-class ties by dict order | [scripts/build_qfabric_labels.py](scripts/build_qfabric_labels.py) | new `_match_classes`: word-boundary regex + earliest-mention order. Behaviourally verified. |

Verification done locally: `py_compile` passes on all edited modules; the matcher fix was
exercised on 5 cases (word-boundary + position-order all correct).

---

## ⚠️ Consequence of the QFabric fix — re-run required

The label-join fix **changes which crops get which labels**, so the committed QFabric numbers in
**§7.8 and §7.9 were computed from mislabelled crops** and must be regenerated on the laptop:
`build_qfabric_labels.py` → `benchmark_qfabric.py` (+`--peft`) → update §7.8/§7.9 tables and
`assets/figures/qfabric_peft_test.png`. Until then, treat those two sections as provisional.

---

## 🖥️ Needs the other laptop (data / GPU / provisioned env)

This machine has **no env at all** (no torch/numpy/pytest; interpreter is Python 3.14, where
some heavy wheels may be unavailable), so nothing below can be done here.

- **Run the fast suite** `pytest -q --ignore=tests/test_text_encoder.py`, then update the stale
  **"129 tests"** claim (REPORT §5 L157, §8 L599, QUICKSTART L142) to the true count
  (static collection is ≥166 — *do not* hand-set a pass count without a real run).
- **Back the §7.5 re-ranking table** with a committed artifact (it currently has none).
- One **real-data seasonal-vs-permanent** quantitative point (or keep the current honest "0 on
  real DEN, demonstrated on fixture" framing — it is already honest).
- Verify the **crop→label timepoint-invariance** assumption (`build` votes on `vids[0]` only).
- Corroborate the **"27,879 labelled crops"** figure with the generated labels JSON.
- Re-run QFabric §7.8/§7.9 (see ⚠️ above).

---

## 🤔 Deferred — your decision (structural or behaviour-changing; not touched)

**Thesis-component readiness (all high — blockers for Q2/Q4 reuse):**
- Package installs under the generic top-level name **`src`** (no `src/__init__.py`; works only
  from repo root) — collides with the thesis repo's own `src`, and the thesis READMEs import a
  name (`open_vocabulary_temporal_change_retrieval`) that doesn't exist. → rename to a unique
  importable package (cross-cutting refactor + thesis-README updates).
- **`requires-python = ">=3.12"`** makes `pip install -e .` impossible in the thesis envs
  (segearthr1 3.10 / lisat 3.9). → lower the floor.
- **`compute_patch_text_similarity` min-max normalises per image to [0,1]** — breaks the
  patch-level CLIP-difference baseline the Q2 plan builds on it (can't difference two
  independently-rescaled maps). → add a raw-cosine patch path (as an encoder extension).
- **`retrieve_changes(query, top_k=5)`** (the Q4 Mode-A tool named in the plan) doesn't exist
  (real entry is `ChangeRetriever.search`). → add a thin wrapper or correct the thesis docs.
- GeoRSCLIP (ViT-B/32) gives only a 7×7 patch grid — coarse for a segmentation baseline.

**Behaviour changes (would alter weak supervision / loaded pixels → need a re-run to assess):**
- `change_type` forced to `"stable"` when dominant class is unchanged even on large sub-dominant
  change; and the degenerate `"X replaced by X"` weak caption for such pairs.
- DEN `.tif` loader min/max-stretches each tile independently → T1/T2 radiometrically
  incomparable for change scoring.

**Protected shared-pipeline files (🔒 fixed doc-side only this session):**
- `src/train.py` default `--split` is `test` + reads an untagged cache; `src/benchmark.py` CLI
  can't find the split-tagged cache. Code-side fixes need your OK to edit protected files; the
  canonical `run_pipeline` path is unaffected.

**Cleanups (done this session):**
- ✅ Loss docs made honest: `train.py` (multi-positive mean over all same-caption positives) vs
  `lora_train.py` (cross-entropy to a single diagonal positive) are no longer called the "same"
  loss (REPORT §7.4 + `lora_train.py` docstring).
- ✅ `InfoNCELoss` in `src/model.py` marked **reference-only** (not the training loss). Deletion
  of it + its tests deferred to a run-capable session (can't re-run the suite here to confirm).
- ✅ `temporal_axis_type='pair'` added to the Protocol docstring's allowed set (`base.py`).
- ✅ `conftest.py` fixture comment corrected (the fixture is committed, not gitignored).
- 📝 Forward-only cross-refs (§7.9 contrast, §10 Limitations citing §7.x): **left as-is** — a
  Limitations section summarising earlier results is standard and readable; revisit only if you
  want strict letter-of-the-rule compliance.

**🆕 New, UNVERIFIED (found while editing — needs a run to confirm):**
- `lora_train._infonce_loss` (lines 101–106) masks with `masked_fill(~(pos_mask | eye), -1e9)`,
  which sets the **true negatives** to −1e9 and keeps only positives+diagonal in the softmax
  denominator — the opposite of masking same-caption *false* negatives. This looks inverted and
  would leave the LoRA loss with no real negatives when a batch has any. LoRA is a secondary
  ablation (§7.4), so it doesn't touch the headline numbers, but verify on the laptop before
  trusting LoRA results. Not changed here (can't test).

**Test gaps:**
- ✅ Color-mode pixel math (NRG band order + NDVI arithmetic + missing-infra fallbacks) —
  **drafted** in `tests/test_dynamic_earthnet_pp.py` (4 tests, pure numpy/PIL, fast-suite). Must be
  **run on the laptop** to confirm (couldn't execute here — no numpy/PIL). The `.npy`
  `DENNpyDataset` end-to-end loader still needs a small fixture to cover.
- 📝 Still to author: `_average_precision` value test on an imperfect ranking; `cache_tag_for`
  tag-string test; `lora_train` smoke test; a direct test of the real `_masked_infonce`;
  `derive_pair_label`.

---

## Full finding inventory (56 confirmed; by dimension)

- **Architecture & invariants** (clean): invariants hold; cache reuse logged & keyed; no
  filename anti-pattern. 📝 `temporal_axis_type='pair'` docstring nit.
- **Case 11 brief**: ✅ every explicit requirement met. 🖥️ seasonal analysis empty on real DEN. ✅ §7.9 figure paraphrase.
- **Claim-vs-evidence**: ✅ §1 numbers, ✅ §7.9 figure, ✅ §7.8 count; 🖥️ §7.5 unbacked table; 🖥️ §8 ops metrics (acceptable, no artifact); 🖥️ "129" count.
- **Datasets & labels**: ✅ QFabric substring + dict-order bugs; 🖥️ timepoint-invariance assumption; 🤔 `change_type`/caption + `.tif` stretch; ✅ subset-after-split disclosure (in §7.2 footnote); 🖥️ 27,879 count.
- **Docs & hygiene** (strong): no secrets; all figures/links resolve. ✅ §7.9 figure; 📝 forward-ref + 12 unreferenced figures.
- **Empirical execution**: 🖥️ env not provisioned (all deps + pytest absent); 🖥️ Python 3.14 vs heavy-wheel availability; positive: `src` imports, `download_den --help` loads; CUDA index correctly platform-gated.
- **Reproducibility**: ✅ `lora_train` default-dataset crash; 🔒 `train`/`benchmark` cache-tag (doc-side fixed); ✅ §9 cache-key wording; ✅ `embeddings` docstring; ✅ duplicate install.
- **Retrieval metrics** (correct math): ✅ cross-split + headline-basis disclosed (§7.2 footnote); 🖥️ seasonal flag 0 on real data; ✅ AP/Recall/baseline verified correct.
- **Scoring** (correct): 📝 train vs lora loss doc mismatch; 📝 dead `InfoNCELoss`.
- **Test coverage**: 🖥️ "129"→true count; 🖥️/📝 untested color-mode math, LoRA path, AP value, `cache_tag`, `derive_pair_label`, dead-class over-testing, no-op assertions; 📝 conftest fixture comment.
- **Thesis-component readiness**: 🤔 package name/`src` collision, `requires-python`, patch normalisation, missing `retrieve_changes`, ViT-B/32 coarse grid (all above).
