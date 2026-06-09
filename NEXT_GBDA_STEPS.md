# Next steps — GBDA Open-Vocabulary Temporal Change Retrieval (living doc)

*Created 2026-06-09.* GBDA-repo only. Merges the **future-work / limitations** from
[`REPORT.md`](REPORT.md) §10 + Appendix B with the **next step opened this session** (the B.13
query-gated hybrid). Live + optional items only; completed/tested work lives in git history +
`REPORT.md`. **Where things run:** doc/CPU work — anywhere; **GPU experiments — RTX 4060 (or a
cloud GPU), not the Mac**; the real Dynamic EarthNet data + caches live on the 4060.

---

## Status (current)

Case 11 deliverable is **complete** — `v1.0`, HuggingFace Space deployed, fast test suite green
(229 passed). The science is fully audited (Appendix B: random baseline + permutation p + BH-FDR +
leakage-free 5-fold CV). Honest headline: the only robust above-random DEN signal is
wetland-formation retrieval; **`patch_top3` is the best config at ~0.193 CV mAP**, and ~0.20 is a
**robust frozen-VLM ceiling**. PEFT/LoRA only memorise train AOIs. The repo is at a natural resting
point — everything below is *optional* beyond the one active item.

---

## Active — the one open item from this session

- [ ] **B.13 query-type-gated global/patch hybrid — run + finalize.** Code + tests + the B.13
  write-up (with a `[PENDING]` results row) are committed-ready; only the run remains.
  - **Run (RTX 4060):** `python -m scripts.patch_eval --encoder georsclip --color-mode nrg --approach gated`
    → `results/patch_eval__georsclip__nrg__gated.json`. Reuses the patch + global caches (no
    re-encode if `data/cache/patch__georsclip__nrg.npz` and the `georsclip…nrg…pair_embeddings`
    split caches exist; else encodes ~1650 images once).
  - **Optional:** repeat `--encoder clip_vitl14` / `remoteclip` for the three-encoder picture.
  - **Then:** fill the B.13 table + per-query FDR verdict from the JSON; commit all four files
    (`src/queries/den.py`, `scripts/patch_eval.py`, `tests/test_patch_gated.py`, `REPORT.md`).
  - **Pre-registered expectation:** recovers the diffuse wetland queries `patch_top3` loses while
    keeping the localised ones → ~5/9 significant, CV mAP 0.19–0.22 — but plausibly within the
    ±0.05 fold variance. Either way it **closes B.11's last open future-work item** (a within-noise
    result is just as reportable as a gain).

---

## Future work (optional research extensions — honest priors)

In rough order of cost/confidence. None is required for the deliverable; the ~0.20 ceiling means
none is likely to move the headline materially.

- [ ] **fMoW as a third dataset** (REPORT §10) — wired only at the protocol level today. Adds
  breadth (functional land-use change over years, global), not DEN performance. File-additive via
  the `TemporalDataset` registry. Labels available **same-source via TEOChatlas** (see *Newer
  dataset options* below) — lower-friction than a fresh download.
- [ ] **QFabric crop-precise (polygon) grounding** (REPORT §10, §7.8–7.10) — the QFabric pipeline
  is label-grounded at crop level; polygon-precise relevance is the remaining QFabric headroom.
- [ ] **Sentinel-1/2 SAR Δ-features** (REPORT §10) — 51/75 AOIs have full S1 coverage; a direct
  extension of the NRG pattern. **High-risk:** CLIP/GeoRSCLIP are optical-pretrained, so SAR↔text
  alignment is dubious; treat as exploratory. *Needs SAR download + a SAR-aware encoder path.*
- [ ] **PEFT generalisation: multi-AOI held-out training / domain-randomised augmentation**
  (REPORT §10, §7.4, B.5) — the one untested mitigation for adapter memorisation. **Pessimistic
  prior:** every learned head on this data has memorised so far; augmentation is the only lever not
  yet tried. *4060/cloud.*
- [ ] **Human relevance judgements** (REPORT §10) — all mAP uses LULC-derived pseudo-labels; human
  query-relevance annotation would give a true IR benchmark. *Annotation effort, not compute.*
- [ ] **Learned global/patch weighting** — only if B.13's a-priori gate shows real signal; a
  trainable per-query weighting is the natural follow-on. **Deferred** by the same memorisation
  risk as the learned attention head (B.12). *4060/cloud.*

### Newer dataset options (online sweep, 2026-06 — from search summaries, not yet verified)

All optional; none is expected to change the ~0.20 frozen-VLM ceiling. No data downloaded.

- [ ] **LEVIR-MCI** — the official LEVIR-CC successor (Change-Agent, IEEE TGRS 2024): the *same*
  10,077 pairs, plus pixel-level change **masks** alongside the 5 captions. Drop-in richer than the
  current `levir_cc` use — adds a localization (mask-IoU) eval on top of caption retrieval, no new
  imagery. arXiv 2403.19646.
- [ ] **Same-source TEOChatlas datasets** — `jirvin16/TEOChatlas` (already parsed for
  `qfabric_teo`/`qfabric_status`) also ships **fMoW** (multi-temporal) and **xBD** + **S2Looking**
  (bi-temporal disaster / general change) in the same RQA instruction format → **same loader
  pattern**, low-friction. Cheapest route to the fMoW item above and to disaster-change. arXiv 2410.06234.
- [ ] **SECOND-CC** (2025) — another human-caption RS change dataset; a 4th open-vocab option for
  the human-relevance angle alongside LEVIR-CC.
- *Related work to cite (not a dataset task):* a 2024 **multimodal RS image change retrieval +
  captioning** framework (arXiv 2406.13424) is the closest published work to this engine; the
  2025–26 open-vocab change-*detection* wave (DynamicEarth, OpenDPR, UniVCD, AdaptOVCD, WHU-GCD) is
  the adjacent frontier. DynamicEarthNet itself has **no successor** — the primary dataset is current.

> **Ruled-out approaches** (already tested — do not re-propose) are documented with results in
> [`REPORT.md`](REPORT.md) Appendix B: equal-weight hybrid + prompt-ensemble (B.11), change-attention
> + learned attention head (B.12), and "more data is the lever" (B.9, refuted).
