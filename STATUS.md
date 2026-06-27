# STATUS — GBDA Case 11 (single source of repo status)

*The one status file. Update **in the same commit** as any state-changing work — never after.
Machine-independent: read after `git pull` on any machine. What physically exists per machine →
[`INVENTORY.md`](INVENTORY.md). Supersedes `NEXT_GBDA_STEPS.md` (folded here 2026-06-10;
completed work lives in git history + [`REPORT.md`](REPORT.md)).*

*Last meaningful update: **2026-06-27** (native-3m controlled JPEG/resolution ablation + temporal-pinpointing study — see §2; QFabric access-channel audit — see §3).*

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
- **Cross-dataset breadth (§7.11/7.13):** LEVIR-CC 5-query — salient construction strong (buildings AP ~0.8, roads ~0.6), subtle/sparse weak (~0.15–0.30), macro ~0.40. SECOND-CC 7-query — every type clears its prevalence floor but modestly (zero-shot macro ~0.33 vs floor 0.30; naive ~0.45 > zero-shot). Same salience law as DEN, wider vocabulary.
- **Localization (§7.12/7.13, LEVIR-MCI + SECOND-CC masks):** the query-conditioned change heatmap is a *weak* localizer — pointing-game lift within ±0.04–0.10 of the random-patch floor; only road on RS-pretrained encoders is clearly positive. Localizing change is harder than retrieving it.
- **Native-3m controlled ablation (§nativeraster, `feature/planet-3m`):** holding AOIs/pairs/colour/encoder/approach/folds fixed and varying *only* image degradation, neither JPEG compression (to q10) nor downsampling (to 64 px) moves retrieval mAP — all rows within one fold std of native (CLIP native 0.130 ± 0.068, max |Δ| 0.036; GeoRSCLIP native 0.085 ± 0.059, max |Δ| 0.027), curves non-monotonic. **Image fidelity is not the bottleneck** — direct evidence for the frozen-VLM method ceiling and vindicates the JPEG-subset data choice. Script `feature_3m_native/jpeg_ablation.py`, fig `jpeg_ablation__*`.
- **Temporal pinpointing (§temporalpinpoint, `feature/planet-3m`):** the brief's "*when* does the change occur" check — rank a AOI's 24 monthly steps by zero-shot Δ-similarity, test whether the score peaks at the true transition month. CLIP ViT-L/14 above chance (macro temporal mAP 0.309 vs 0.207 random; peak within ±1 month 63%), strongest for sharp events (snow melt 0.586, p=0.025); GeoRSCLIP ViT-B/32 at/below chance (0.146 vs 0.208) — **encoder-dependent**. Small per-query AOI counts (1–7) → 0/5 clears BH-FDR. **Pinpointing *when* (like *where*) is harder than retrieving *whether*.** Script `feature_3m_native/temporal_pinpoint.py`, fig `temporal_pinpoint__*`.
- Ruled-out approaches documented in REPORT Appendix B (B.9, B.11, B.12) — do not re-propose.
- Engine **deployed on HF Space**; full test suite **passes** (1 skipped: the real-CLIP `test_text_encoder`, needs weights; ~90 s, shared venv) — re-run `pytest -q` after pulling to confirm the current count.

## 3. Running now

Nothing executing. **Engineering (2026-06-17):** merged `fix/gbda-audit` into `main` (FF) —
security hardening (path-traversal-safe archive extraction M2; app binds `127.0.0.1` by default,
auto `0.0.0.0` on a HF Space M3), encoder-agnostic LoRA seam + HF-CLIP LoRA support (H2),
batch/reuse patch embeddings (M15/M16), split/colour-keyed embedding caches, dead-code removal
(`InfoNCELoss` + unused feature helpers), and `QUICKSTART.md`. Suite green; re-run `pytest -q` to
confirm the count. App UX pass: collapsible About now covers open-vocabulary, the PEFT honesty
note, and the seasonal/permanent result note; NRG/NDVI greyed out on non-DEN corpora.

**Status (2026-06-13):** the data-expansion + honest-reframe plan
([`docs/DATA_EXPANSION_PLAN.md`](docs/DATA_EXPANSION_PLAN.md)) is **complete through Tracks 0–4**;
the only remaining item, QFabric, is blocked on external access (below). **Done:**
Track 0 (disk audit + cleanup — ~14 GB reclaimed across both repos, now ~55 GB free), Track 1
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
LEVIR-MCI). REPORT §7.13. **Track 4 DONE** — anti-memorization check
(`scripts/peft_augment_eval.py`): embedding-space feature-noise does not help PEFT (σ=0.25 → 0.206
within noise of no-aug 0.196; degrades higher). It also surfaced + fixed a mis-cited claim — matched
leakage-free CV PEFT (NRG 0.196 ± 0.049) *overlaps* frozen zero-shot (0.139 ± 0.024), not the RGB
0.049 some summaries paired against NRG (REPORT B.14; all docs aligned). **QFabric (the last
remaining expansion item) is now DROPPED from scope (2026-06-27)** — the full 5-date polygon-mask
form is unobtainable: the `labaerien/qfabric` HF mirror stayed gated with manual review pending
~2 weeks, and **every contact channel is dead** (`hello@labaerien.com` bounces — GitHub-Pages
domain, no mail server; `sagar@granular.ai` 550 "account inactive" — Granular folded, lead author
left RS; `engine.granular.ai` Cloudflare 1016 origin-DNS). No deliverable impact: Tracks 0–4
already satisfy every assignment requirement, so the QFabric localization slice was a *bonus*, not a
gate. The existing reduced `qfabric_teo` (2-date retrieval crops) stays in the corpus. Channel-audit
detail + (now-stale) request drafts archived at `data/_notes/qfabric_access_drafts.md` (untracked);
full historical spec at [`docs/QFABRIC_FUTURE_WORK.md`](docs/QFABRIC_FUTURE_WORK.md) (marked DROPPED).

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
5. `[optional]` UX items — [`docs/UX_DESIGN.md`](docs/UX_DESIGN.md). **Instant search via precomputed
   embeddings DONE 2026-06-20** (the highest-leverage item): per-patch embeddings now persist to an
   on-disk cache (`PatchEmbeddingStore` + `load_or_compute_patches` in `src/embeddings.py`, keyed by
   the same `cache_tag_for` (split, colour, lora) as the pair store and pair-order-guarded). `app.py`
   `_patch_scores` loads the warm cache instead of re-encoding on the first `approach="patch"` query
   (DEN georsclip/nrg: ~11 s GPU pass → 0.076 s warm load, scores identical); warm the cache offline
   via `scripts/precompute_patch_embeddings.py`. Tests `tests/test_patch_cache.py` (3, round-trip +
   reuse + stale-pairset guard); suite 234 green. Remaining optional UX = cross-dataset fusion search
   (Idea 2).

## 5. Future path (priors honest — see [`docs/DATA_EXPANSION_PLAN.md`](docs/DATA_EXPANSION_PLAN.md) for the committed set)

**Reassessed 2026-06-12.** Root cause of the weak numbers is **purpose-mismatch + method ceiling**,
not subset size: DEN was used in full (75 AOIs, dense labels) and is source-maxed; QFabric was used
in a reduced TEOChatlas form (no masks, 2/5 dates); LEVIR-CC was under-utilized (now broadened to
5 queries: salient construction strong — buildings AP ~0.8, roads ~0.6 — subtle/sparse weak —
demolition/vegetation/water ~0.15–0.30; 5-query macro ~0.40, was a 3-query ~0.55 carried by buildings).

- **Committed (in the plan), all DONE:** LEVIR-MCI (same images + building/road masks → localization
  eval), SECOND-CC (30 change categories → open-vocab breadth), DEN finer (monthly) temporal pairing.
  Localization/heatmap eval is now a first-class pillar.
- **Dropped 2026-06-27:** QFabric pentatemporal + polygon masks — full mask form unobtainable (HF
  `labaerien/qfabric` gated + every contact channel dead; see §3). Reduced `qfabric_teo` (2-date
  retrieval) stays. Localization pillar already covered by LEVIR-MCI + SECOND-CC.
- **Rejected with reasons:** **fMoW** — functional-classification dataset with **no change labels**;
  repurposing reintroduces the weak-label noise that sank DEN PEFT (rationale in the plan §2). DEN
  alternative sources (TUM raw / HEVC / torchgeo) — same labels, no gain. **OSCD**
  ([rcdaudt.github.io/oscd](https://rcdaudt.github.io/oscd/), live) — evaluated 2026-06-27, rejected:
  **binary** urban change only (no semantic classes → no open-vocab vocabulary), Sentinel-2 10–60 m
  (coarse → domain shift for the high-res-trained CLIP variants), 24 pairs, 2 dates. Its pixel masks
  only support *binary* localization, already covered better by high-res class-labelled LEVIR-MCI +
  SECOND-CC. The "OSCD + Our Dates" multidate extension + Siamese weights are on the dead Granular
  engine (CF 1016). Off-brief, strictly weaker than the existing corpus — do not integrate.
- **Still future-work:** S1 SAR Δ-features (off-brief, optical-encoder mismatch) · PEFT
  anti-memorisation via augmentation · human relevance judgements (annotation, no compute).

Related-work cite: arXiv 2406.13424 + the 2025–26 open-vocab change-detection wave.
