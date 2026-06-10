# STATUS — GBDA Case 11 (single source of repo status)

*The one status file. Update **in the same commit** as any state-changing work — never after.
Machine-independent: read after `git pull` on any machine. What physically exists per machine →
[`INVENTORY.md`](INVENTORY.md). Supersedes `NEXT_GBDA_STEPS.md` (folded here 2026-06-10;
completed work lives in git history + [`REPORT.md`](REPORT.md)).*

*Last meaningful update: **2026-06-10**.*

---

## 1. Purpose & scope

**GBDA course lab project — Case 11: Open-Vocabulary Temporal Change Retrieval** (assignment:
[`docs/GBDA_Case11_Overview.md`](docs/GBDA_Case11_Overview.md)). Natural-language queries over
multitemporal RS tiles → ranked change events with side-by-side pairs + heatmaps + confidence.
Frozen CLIP-variant backbones, zero-shot + light PEFT, retrieval metrics + seasonal-drift error
analysis, Gradio deliverable. **2-month course project, 2 students** (second teammate's joining
mode TBD — onboarding docs written to work either way).

**Runtime policy (hard):** RTX 4060 laptop = primary compute. **NO ARIS** (ARIS access is granted
for the thesis, not this course; the GBDA clone on ARIS exists solely as a thesis *library* —
no course jobs there). **No MacBook hardware runs** (docs/light dev only). Colab/Kaggle = legal
burst capacity if 8 GB VRAM short. Thesis ↔ GBDA: thesis pins GBDA `v1.0` as a library; ideas may
flow back, thesis workload does not.

## 2. Results state (canonical — REPORT.md @ HEAD)

Honest, audited numbers (random baseline + permutation p + BH-FDR + leakage-free 5-fold CV):

- **Best config: GeoRSCLIP + NRG `patch_top3` — CV mAP 0.193 ± 0.051, 4/9 queries FDR-significant**
  (buildings, urban, wetland↔farmland transitions). **~0.20 is a robust frozen-VLM ceiling.**
- B.13 query-gated hybrid 0.186 ± 0.051 (4/9) — within noise; geometry routing doesn't help (closed).
- Global zero-shot 0.139 ± 0.024 (2/9); full-corpus macro 0.116; **0.426 = lucky single fold, never cite**.
- PEFT/LoRA memorise train AOIs — zero-shot generalises better (B.5); seasonal gate FPR→0 at thr ≥ 0.05.
- Ruled-out approaches documented in REPORT Appendix B (B.9, B.11, B.12) — do not re-propose.
- Engine **deployed on HF Space**; tests 233 passed (fast suite ~2 min, shared venv).

## 3. Running now

Nothing. Repo at a natural resting point; science complete for the deliverable.

## 4. Next steps — course-deliverable gap pass (deliverable = written report + repo with presentation-grade README)

Mapping the assignment + professors' 2026-05-06 comments to repo reality:

| requirement | state |
|---|---|
| subset of datasets | ✅ DEN (primary) + QFabric + LEVIR-CC |
| CLIP-variant embeddings | ✅ CLIP ViT-L/14, GeoRSCLIP, RemoteCLIP (3-encoder comparison) |
| zero-shot & light PEFT | ✅ both run; honest leakage-free comparison (PEFT memorises) |
| visual comparisons zero-shot vs PEFT | ⚠ verify a dedicated figure exists; make one if not (CV/S1/S3 progression figure exists; ZS-vs-PEFT side-by-side TBC) |
| retrieval metrics (mAP/R@K) + temporal pinpointing | ✅ REPORT §7 + B (R@K ceiling-bounded caveat documented) |
| seasonal-drift error analysis | ✅ seasonal gate + stable-pair FPR analysis |
| Gradio semantic change search engine | ✅ deployed (HF Space): ranked list, T1/T2 side-by-side, heatmap, score |
| **written report** | 🔄 `main.tex` **now tracked** (user decision 2026-06-10; un-ignored) — next: align it to REPORT.md canon numbers |
| **README as presentation** | ⚠ upgrade to presentation-grade: results table, screenshots/GIF of the app, quickstart, module map, honest-headline framing |

Ordered:
1. Align the now-tracked `main.tex` to REPORT.md canon numbers (home decided 2026-06-10: tracked in repo).
2. README presentation-grade pass (app screenshots need one render session).
3. Verify/make the zero-shot-vs-PEFT visual comparison figure.
4. Teammate onboarding once joining mode decided (`STATUS.md` + README are the onboarding).
5. `[optional]` UX items (instant search via precomputed embeddings is the highest-leverage one) — [`docs/UX_DESIGN.md`](docs/UX_DESIGN.md).

## 5. Future path (optional research extensions — honest priors; none required, ~0.20 ceiling unlikely to move)

fMoW via same-source TEOChatlas (lowest friction) · QFabric polygon-precise grounding · S1 SAR
Δ-features (high-risk: optical-pretrained encoders) · PEFT anti-memorisation via augmentation
(last untested lever) · human relevance judgements (annotation, no compute) · LEVIR-MCI masks
(adds localization eval) · SECOND-CC. Related-work cite: arXiv 2406.13424 + the 2025–26 open-vocab
change-detection wave. DEN has no successor — primary dataset is current.
