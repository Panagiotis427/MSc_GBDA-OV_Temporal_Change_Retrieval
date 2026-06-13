# STATUS — GBDA Case 11 (single source of repo status)

*The one status file. Update **in the same commit** as any state-changing work — never after.
Machine-independent: read after `git pull` on any machine. What physically exists per machine →
[`INVENTORY.md`](INVENTORY.md). Supersedes `NEXT_GBDA_STEPS.md` (folded here 2026-06-10;
completed work lives in git history + [`REPORT.md`](REPORT.md)).*

*Last meaningful update: **2026-06-12**.*

---

## 1. Purpose & scope

**GBDA course lab project — Case 11: Open-Vocabulary Temporal Change Retrieval** (assignment:
[`docs/GBDA_Case11_Overview.md`](docs/GBDA_Case11_Overview.md)). Natural-language queries over
multitemporal RS tiles → ranked change events with side-by-side pairs + heatmaps + confidence.
Frozen CLIP-variant backbones, zero-shot + light PEFT, retrieval metrics + seasonal-drift error
analysis, Gradio deliverable. **2-month course project, 2 students** (second teammate's joining
mode TBD — onboarding docs written to work either way).

**Runtime policy (hard):** RTX 4060 laptop = primary compute. **No MacBook hardware runs**
(docs/light dev only). Colab/Kaggle = legal burst capacity if 8 GB VRAM short.

## 2. Results state (canonical — REPORT.md @ HEAD)

Honest, audited numbers (random baseline + permutation p + BH-FDR + leakage-free 5-fold CV):

- **Best config: GeoRSCLIP + NRG `patch_top3` — CV mAP 0.193 ± 0.051, 4/9 queries FDR-significant**
  (buildings, urban, wetland↔farmland transitions). **~0.20 is a robust frozen-VLM ceiling.**
- B.13 query-gated hybrid 0.186 ± 0.051 (4/9) — within noise; geometry routing doesn't help (closed).
- Global zero-shot 0.139 ± 0.024 (2/9); full-corpus macro 0.116; **0.426 = lucky single fold, never cite**.
- PEFT: high in-distribution train-fit is memorisation; leakage-free CV PEFT (NRG 0.196 ± 0.049) overlaps frozen zero-shot (0.139 ± 0.024) within fold variance — no clear OOD gain, not a collapse (B.5/B.9/B.14; RGB lower at 0.049). Feature-noise augmentation doesn't help (B.14). Seasonal gate FPR→0 at thr ≥ 0.05.
- Ruled-out approaches documented in REPORT Appendix B (B.9, B.11, B.12) — do not re-propose.
- Engine **deployed on HF Space**; tests 233 passed (fast suite ~2 min, shared venv).

## 3. Running now

Nothing executing. **Active direction (2026-06-12):** the data-expansion + honest-reframe plan
([`docs/DATA_EXPANSION_PLAN.md`](docs/DATA_EXPANSION_PLAN.md)) is in implementation. **Done:**
Track 0 (disk audit — 56.6 GB free, ~4 GB redundant archives reclaimed, gate PASS), Track 1
(honest reframe — verified, zero stale `0.426` headlines), and the Track 2 **LEVIR-CC 5-query
broadening** (added vegetation + water queries → salience gradient: buildings ~0.8, roads ~0.6,
demolition/vegetation/water ~0.15–0.30; macro ~0.40; docs reframed; 9 tests green). **Next:**
**Track 3 (LEVIR-MCI localization) DONE** — downloaded LEVIR-MCI (2.77 GB, building/road masks on
the same 1929 test pairs), added `levir_mci` loader + `scripts/eval_localization.py`. Honest
negative: the query-conditioned change heatmap is a weak localizer — only road localizes above
chance and only for RS-pretrained encoders (RemoteCLIP +0.10 / GeoRSCLIP +0.08 pointing lift);
building at/below floor for all; generic CLIP anti-localizes (REPORT §7.12, main.tex §localization).
**DEN-monthly DONE** — `patch_eval.py --pairing monthly` (additive; bimonthly cache untouched):
GeoRSCLIP NRG patch_top3 CV mAP 0.138 ± 0.046 (1725 pairs) vs bimonthly 0.193 — honest granularity
tradeoff (finer pairs, less per-pair change, 2× temporal resolution; wetland signal stable). REPORT
B.10. **SECOND-CC DONE** — downloaded the real SECOND-CC (Zenodo `10.5281/zenodo.16937571`, 2.5 GB,
public, CC-BY-4.0; the captioned superset, **not** the captionless SECOND base): 6,041 pairs +
30,205 human captions + six-class semantic maps. Added `second_cc` loader + 7 class queries +
`benchmark_second_cc.py`; localization via the generalized `eval_localization.py --dataset`. The
open-vocab **breadth** test: across 7 change types every query clears its prevalence floor (unlike
DEN) but modestly — zero-shot macro ~0.33 / naive ~0.45 vs floor 0.30; buildings ~0.70 dominate,
water/playground weak; naive > zero-shot (end-state > Δ). Localization weak (lifts ±0.04, matches
LEVIR-MCI). REPORT §7.13. **Next (BLOCKED on external access):** QFabric pentatemporal — needs
Granular login or a capped extraction; deferred pending access.

> **⚠ On `laptop-4060`, before ANY dataset download:** disk-gated. Track 0 cleared the gate
> (56.6 GB free after reclaiming `labels.tar.gz` + `Levir-CC-dataset.zip` redundant archives to
> `trash/`; ~30 GB worst-case spare). Still **never** pull the 298 GB EVER-Z QFabric parquet; cap
> the QFabric slice. Full budget: [`docs/DATA_EXPANSION_PLAN.md §3`](docs/DATA_EXPANSION_PLAN.md)
> · [`INVENTORY.md`](INVENTORY.md).

## 4. Next steps — course-deliverable gap pass (deliverable = written report + repo with presentation-grade README)

Mapping the assignment + professors' 2026-05-06 comments to repo reality:

| requirement | state |
|---|---|
| subset of datasets | ✅ DEN (primary) + QFabric + LEVIR-CC + LEVIR-MCI (masks) + SECOND-CC (breadth, masks) |
| CLIP-variant embeddings | ✅ CLIP ViT-L/14, GeoRSCLIP, RemoteCLIP (3-encoder comparison) |
| zero-shot & light PEFT | ✅ both run; honest leakage-free CV (PEFT ≈ frozen zero-shot OOD within variance; train-fit is memorisation) |
| visual comparisons zero-shot vs PEFT | ✅ `assets/figures/zeroshot_vs_peft__clip_vitl14__train.png` — embedded in `main.tex` (§6.1) and README; generator `scripts/make_comparison_figure.py` |
| retrieval metrics (mAP/R@K) + temporal pinpointing | ✅ REPORT §7 + B (R@K ceiling-bounded caveat documented) |
| seasonal-drift error analysis | ✅ seasonal gate + stable-pair FPR analysis |
| Gradio semantic change search engine | ✅ deployed (HF Space): ranked list, T1/T2 side-by-side, heatmap, score |
| **written report** | ✅ `main.tex` tracked + **aligned to REPORT canon (2026-06-10)**: abstract carries 0.100 ± 0.139 and 0.193 ± 0.051 exactly; B.13 gated-routing null result added to the honest-negatives paragraph; pure formal prose (zero list blocks / Q&A patterns), zero style violations |
| **README as presentation** | ✅ presentation-grade (2026-06-10): demo screenshot + "Results at a glance" (audited canon table + cv_progression + zero-shot-vs-PEFT figures) + quickstart + module map + honest-headline framing |

Ordered:
1. ~~Align `main.tex` to REPORT canon~~ **DONE 2026-06-10** (exact CV intervals in abstract; B.13 null result added).
2. ~~README presentation-grade pass~~ **DONE 2026-06-10** ("Results at a glance" + figures; screenshot already present).
3. ~~Zero-shot-vs-PEFT comparison figure~~ **DONE — verified existing + embedded** (main.tex §6.1, README).
4. Teammate onboarding once joining mode decided (`STATUS.md` + README are the onboarding).

*Repo restructure 2026-06-10: inventory tooling moved to `ops/` (`make_inventory.ps1/.sh`);
all written references repointed.*
5. `[optional]` UX items (instant search via precomputed embeddings is the highest-leverage one) — [`docs/UX_DESIGN.md`](docs/UX_DESIGN.md).

## 5. Future path (priors honest — see [`docs/DATA_EXPANSION_PLAN.md`](docs/DATA_EXPANSION_PLAN.md) for the committed set)

**Reassessed 2026-06-12.** Root cause of the weak numbers is **purpose-mismatch + method ceiling**,
not subset size: DEN was used in full (75 AOIs, dense labels) and is source-maxed; QFabric was used
in a reduced TEOChatlas form (no masks, 2/5 dates); LEVIR-CC was under-utilized (now broadened to
5 queries: salient construction strong — buildings AP ~0.8, roads ~0.6 — subtle/sparse weak —
demolition/vegetation/water ~0.15–0.30; 5-query macro ~0.40, was a 3-query ~0.55 carried by buildings).

- **Committed (in the plan):** LEVIR-MCI (same images + building/road masks → localization eval),
  SECOND-CC (30 change categories → open-vocab breadth), QFabric pentatemporal + polygon masks
  (disk-gated localization upgrade — never the 298 GB EVER-Z parquet), DEN finer (monthly) temporal
  pairing. Localization/heatmap eval is now a first-class pillar.
- **Rejected with reasons:** **fMoW** — functional-classification dataset with **no change labels**;
  repurposing reintroduces the weak-label noise that sank DEN PEFT (rationale in the plan §2). DEN
  alternative sources (TUM raw / HEVC / torchgeo) — same labels, no gain.
- **Still future-work:** S1 SAR Δ-features (off-brief, optical-encoder mismatch) · PEFT
  anti-memorisation via augmentation · human relevance judgements (annotation, no compute).

Related-work cite: arXiv 2406.13424 + the 2025–26 open-vocab change-detection wave.
