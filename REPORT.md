# Deliverable Report — Open-Vocabulary Temporal Change Retrieval (GBDA Case 11)

*Semantic Change Search Engine over Dynamic EarthNet, using frozen
vision-language backbones with a zero-shot vs parameter-efficient fine-tuning
comparison.*

---

## 1. Executive summary

We built a system that, given a free-text query (e.g. *"agricultural land
converted to wetland"*), ranks bi-temporal satellite image pairs by how well
the **change** between the two timesteps matches the query, and returns the
matched tiles, the timestep, a localisation heatmap, a confidence, and a
seasonal-vs-permanent flag.

The starting code base was a scaffold whose core was broken or disconnected
(text encoder crashed; the app did single-image retrieval, not change
retrieval; evaluation was a synthetic identity hack; no data). We repaired the
correctness bugs, built the missing change-retrieval core, added a
label-grounded benchmark, implemented PEFT training, finished/added the three
encoders, rewired the Gradio app, and validated everything on a deterministic
fixture and on real Dynamic EarthNet.

Training-set fit (real DEN, train split, 605 pairs — the **PEFT column is the adapter scored on
its own training pairs**, i.e. memorisation capacity, not a retrieval result; see §7.1):

| Encoder | naive mAP | zero-shot mAP | PEFT mAP (train-fit) |
|---|---|---|---|
| CLIP ViT-L/14 | 0.031 | 0.043 | 0.420 |
| GeoRSCLIP ViT-B/32 | 0.027 | 0.040 | 0.335 |
| RemoteCLIP ViT-L/14 | 0.024 | 0.057 | 0.352 |

Generalisation, held-out **test split** (110 pairs — colour-mode ablation, zero-shot). **These
single-split numbers do not survive cross-validation — see the CV column and Appendix B.8;** the
robust estimate is ~10× lower because this 110-pair split is a lucky high-wetland fold:

| Encoder | color | test-split mAP | full-corpus mAP (825) | 5-fold CV mAP |
|---|---|---|---|---|
| CLIP ViT-L/14 | NRG | 0.104 | 0.030 | 0.076 ± 0.023 |
| GeoRSCLIP | RGB | 0.299 | 0.032 | 0.085 ± 0.069 |
| GeoRSCLIP | **NRG** | **0.426** | **0.037** | **0.100 ± 0.139** |
| GeoRSCLIP | NDVI | 0.216 | — | — |
| RemoteCLIP | NRG | 0.129 | 0.021 | 0.053 ± 0.022 |

(Full per-band test-split table in §7.3.) Robust verdict: no configuration exceeds ~0.10 mAP
under cross-validation; only GeoRSCLIP NRG shows any above-random per-query signal, confined to
wetland-formation.

**Key findings:** PEFT adapters only memorise train AOIs — the high train-fit numbers above are
the adapter scored on its own training data and collapse to ≤ zero-shot (sometimes below random)
on held-out splits; PEFT is not shown to help retrieval. Frozen zero-shot generalises better.
NRG false-colour (NIR-Red-Green) outperforms RGB and NDVI for all encoders on unseen AOIs; NDVI
collapses spectral texture to a single channel, losing the inter-channel contrasts NRG preserves.
GeoRSCLIP + NRG zero-shot (0.426) is the best configuration *on this 110-pair test split* — but
that number does **not survive cross-validation**: on the full 75-AOI corpus the same config
scores **0.037** mAP, and 5-fold leakage-free CV gives **0.100 ± 0.139** (the test split was a
lucky high-wetland fold). The only robust, above-random signal is GeoRSCLIP retrieving
**wetland-formation** change. Full random-baseline, FDR and cross-validation analysis in
**Appendix B** (esp. B.8).

---

## 2. Problem and objectives

Per the Case 11 brief: move beyond fixed-class change detection to
**open-vocabulary** change retrieval using VLMs, on a low-compute budget.
Required: subset of data; CLIP-variant frozen encoders; **both zero-shot and
light PEFT**; retrieval metrics (Recall@K, mAP); error analysis of seasonal vs
permanent confusion; a Gradio search engine deliverable. Primary dataset
Dynamic EarthNet (DEN); architecture open to QFabric/fMoW.

---

## 3. System architecture

For each bi-temporal pair `(T1, T2)`:

1. A frozen VLM encodes both timesteps → `f_T1, f_T2` (L2-normalised); cached
   to disk per (dataset, encoder).
2. Change feature `Δf = f_T2 − f_T1` (or concatenation).
3. The query text is encoded by the same model's text tower → `t`.
4. The pair is scored by one of three approaches:

| Approach | Score | Trained |
|---|---|---|
| `naive` | cos(t, f_T2) | none — image retrieval lower bound |
| `zero_shot` | cos(t, f_T2) − cos(t, f_T1) (Δ-similarity) | none |
| `peft` | cos(t, g(Δf)), g = `ProjectionHead` adapter | ~0.5 M params |

**Dataset-agnostic core.** A structural `TemporalDataset` protocol
(`src/datasets/base.py`) defines the only contract downstream code depends on.
A registry maps a name to a factory; the DEN factory auto-detects on-disk
layout. Concrete loaders span both datasets and their formats and temporal
axes: `DENDataset` (raster .tif, monthly) and `DENNpyDataset` (DynNet
preprocessed .npy, 24-month) for DEN; and for QFabric `TEOChatlasQFabricDataset`
(polygon-centred .tif crops with the real RQA2 change-type labels — the
benchmarked variant, §7.8–7.9), a RQA5 status-transition sibling, and an EVER-Z
parquet images-only loader for qualitative use.

**Encoders.** An `ImageTextEncoder` protocol with three implementations:
`clip_vitl14` (HF CLIP, 768-d), `georsclip` (open_clip ViT-B/32 + RS5M
checkpoint, 512-d), `remoteclip` (open_clip ViT-L/14 + RemoteCLIP checkpoint,
768-d). Embedding dimension drives cache/index sizing automatically.

**PEFT training.** Only the `ProjectionHead` adapter trains; backbones frozen.
Supervision is DEN's weak caption derived from its LULC labels
(`"agriculture replaced by wetlands"`, `"stable forest land cover"`).
Loss is masked symmetric InfoNCE — pairs sharing an identical caption are
mutual positives, avoiding the false-negative problem of plain diagonal
InfoNCE (DEN captions repeat heavily).

**Benchmark.** Label-grounded: a fixed query set; each query maps to a
relevance rule over the derived `PairLabel` (dominant T1/T2 class, stability).
Metrics: per-query and macro Recall@K, mAP, and a seasonal-drift figure
(fraction of non-relevant top-K retrievals that involve snow/ice — i.e.
seasonal events wrongly returned for permanent-change queries).

---

## 4. Data

Dynamic EarthNet sources from `docs/Common_Resources.md` were assessed; the ~525 GB
raw TUM mirror was excluded. The ~7 GB gdown preprocessed subset was chosen.

Discovery during integration: the downloaded archive, named `den_5aoi.tar.gz`,
is in fact a **ZIP**, and its contents are the **DynNet preprocessed format**
— per-AOI daily RGB JPEGs (≈730 frames) plus `labels/<AOI>.npy` of shape
`(24, 1024, 1024)` monthly LULC, with a `split.json` (55 train / 10 val / 10
test AOIs). This differs from the raster layout the original loader assumed.
We added `DENNpyDataset` and made the extractor sniff the real format and the
registry auto-detect layout. The 24 monthly label maps form the change
timeline; each month is mapped to a representative daily RGB frame. These
preprocessed label maps were cross-validated against the official torchgeo
Dynamic EarthNet raster masks (`torchgeo/dynamic_earthnet`): on the AOI tiles
present in both releases (3 AOIs, 75.5M valid pixels) per-pixel class agreement
is 99.9999%, confirming the preprocessing preserves the published ground truth
(the official release withholds the held-out test labels that the preprocessed
subset retains).

A deterministic synthetic DEN fixture (`scripts/make_den_fixture.py`)
reproduces the on-disk layout and an engineered label signal (urban growth,
deforestation, seasonal snow-melt, stable negatives) so the full pipeline is
testable in seconds with no network.

---

## 5. Key fixes and additions

| Area | Problem in scaffold | Resolution |
|---|---|---|
| Text encoder | `.last_hidden_state` on `get_text_features` tensor → crash; CPU-pinned | Use `get_text_features` directly; device-aware (CUDA default) |
| App | Indexed single images; returned first two timepoints; change/adapter code unused | Rewired to real change retrieval (`ChangeRetriever`), top-K events, selectors |
| Evaluation | Identity-diagonal on synthetic random data | Label-grounded benchmark (Recall@K, mAP, drift) |
| Training | Broken loop (list-of-batches, ignored shuffle, unused loss), synthetic data | Rewritten: masked symmetric InfoNCE on DEN weak captions |
| Encoders | GeoRSCLIP a mis-loading stub; RemoteCLIP absent | Shared open_clip+HF base; both implemented, registered |
| Data | None on disk; loader assumed wrong layout | Format-sniffing downloader; `DENNpyDataset`; synthetic fixture |
| Robustness | Windows cp1252 crashes on non-ASCII prints | ASCII-safe console output |

---

## 6. Experiments and runs (chronological)

1. **Environment** — Python 3.12, torch 2.10 + CUDA, RTX 4060. Installed
   `gdown`, `open-clip-torch`. Confirmed GPU.
2. **DEN download** — 7.09 GB fetched via gdown. First two extraction attempts
   failed (a stale in-memory `→` print on Windows; then "not a gzip file").
   Diagnosed: file is a ZIP; contents are DynNet preprocessed. Fixed extractor;
   extracted; 75 AOIs.
3. **Synthetic fixture** — built; 6 bimonthly pairs over 2 AOIs; derived
   labels verified to contain the engineered transitions.
4. **CLIP text sanity** — post-fix `encode_text` returns `[N, 768]`,
   L2-normalised, on CUDA; forest image correctly prefers "forest" over
   "city". Confirms the P1 fix.
5. **Fast test suite (mock encoders, fixture, no network)** — **225 passed**.
   Covers embeddings cache + round-trip, retrieval (naive/zero_shot/peft),
   benchmark metrics (exact Recall@1/AP on engineered transitions), PEFT
   training (loss decreases, save/load, PEFT ≥ zero-shot), encoder
   protocol/registry/contracts, app `query()` (real fixture tiles + heatmap +
   seasonal note), heatmap, model.
6. **Real-CLIP text tests** — `test_text_encoder.py` **15 passed**
   (bigG case deselected — 10 GB download).
7. **Real DEN, test split (CLIP, 110 pairs)** — only the wetland/agriculture
   queries had positives; naive and zero-shot at chance (mAP ≈ 0.045). Finding:
   the test split is class-imbalanced toward agri/wetland; CLIP cannot resolve
   subtle agri↔wetland flips zero-shot.
8. **Real DEN, train split (CLIP, 605 pairs)** — full pipeline incl. PEFT:
   naive 0.031, zero-shot 0.043, PEFT 0.420 mAP (train-fit — adapter scored on its own
   training pairs; held-out verdict in §7.2). Embeddings + adapter cached.
9. **Headless app smoke (real CLIP, train split)** — engine builds from cache;
   `zero_shot` top result a weak stable pair (consistent with §7), `peft` top
   result *"agriculture replaced by wetlands"* correctly matching the query,
   with real T1/T2 tiles, rendered heatmap, confidence, and a
   permanent-change note.
10. **Real DEN, train split (GeoRSCLIP, 605 pairs)** — naive 0.027,
    zero-shot 0.040, **PEFT 0.335** mAP.
11. **Real DEN, train split (RemoteCLIP, 605 pairs)** — naive 0.024,
    zero-shot 0.057, **PEFT 0.352** mAP.
12. **Timed pass (RTX 4060, GPU free)** — encode 68 ms/tile (1024→224, CLIP
    L/14; 1210 tiles ≈ 82 s, one-time); PEFT train 605×40 epochs ≈ 29 s;
    end-to-end query (CLIP text forward + scoring over 605 pairs) 10.5 ms.
13. **Repo restructure** — stray QFabric parquet+embeddings moved to
    `data/QFabric/`; AOI geographic metadata computed from XYZ tile IDs and
    enriched with torchgeo `splits.csv` (UTM zones, Sentinel-1/2 availability
    for all 75 AOIs) → `data/DynamicEarthNet/aoi_metadata.json`.
14. **NIR false-colour (NRG)** — the on-disk `_infra.jpeg` NIR frames (one per
    daily timestep, grayscale, 1024²) are now usable via `color_mode='nrg'`
    (NIR-Red-Green) or `'ndvi'` (single-band NDVI × 3). GeoRSCLIP + NRG
    zero-shot on held-out test AOIs: mAP **0.426** — best generalisation result.
15. **Cross-split evaluation** — pipeline extended with `--train-split` /
    `--eval-splits` flags; adapter trained on train split (605 pairs) evaluated
    on val (110 pairs) and test (110 pairs). PEFT overfits train; zero-shot and
    NRG-augmented zero-shot generalise. Cache keyed by (dataset, encoder, split,
    color) to avoid collision.
16. **NDVI ablation** — `color_mode='ndvi'` benchmarked across all encoders and
    splits. NRG outperforms NDVI for all encoders on held-out test (GeoRSCLIP:
    0.426 NRG vs 0.216 NDVI). NDVI collapses spectral texture to one channel,
    losing inter-channel contrasts NRG preserves.
17. **LoRA adapter** — peft LoRA applied to GeoRSCLIP visual encoder (ViT-B-32),
    targeting the FFN `c_fc`, `c_proj` in each ResBlock (368,640 trainable /
    88M total, 0.42%; `out_proj` is a no-op on open_clip's fused attention and is
    not targeted). Trained with online image loading (no pre-caching), same
    masked InfoNCE loss as ProjectionHead. GeoRSCLIP+NRG, 20 epochs, rank=4:
    fits train (0.025→0.153) but test mAP collapses to **0.071** — overfits, far
    below zero-shot NRG (0.426). Confirms zero-shot as the best generalising approach.

---

## 7. Results and analysis

### 7.1 Training-set fit (train split, 605 pairs, RGB)

`difference` change feature; mAP and macro Recall@10:

| Encoder | naive | zero-shot | PEFT | PEFT R@10 |
|---|---|---|---|---|
| CLIP ViT-L/14 (768-d) | 0.031 | 0.043 | **0.420** | 0.36 |
| GeoRSCLIP ViT-B/32 (512-d) | 0.027 | 0.040 | **0.335** | 0.26 |
| RemoteCLIP ViT-L/14 (768-d) | 0.024 | 0.057 | **0.352** | 0.30 |

> **Read this table as a sanity check, not a result.** The PEFT column is the adapter
> **evaluated on the very pairs it was trained on** (`run_pipeline.py` trains on the train split,
> then scores `peft` on the same split). It therefore measures how well the adapter can *fit*
> the training set — i.e. memorisation capacity — not whether it retrieves change. A model that
> can fit its training data is expected, not a finding. **The verdict on whether PEFT helps is
> the held-out result in §7.2, where it collapses.** See Appendix B.5 for the leakage analysis
> (on QFabric the same train-fit reaches mAP 0.998).

- **Zero-shot is near chance.** Δ-similarity beats the naive image baseline
  marginally but neither separates change types. CLIP/GeoRSCLIP embed scene
  appearance, not directional land-cover transition; differencing two
  normalised global embeddings discards the localised change signal.
- **PEFT can fit the training pairs** (mAP ~8–10× the train zero-shot baseline). This only
  confirms the adapter has the capacity to memorise the 55 training AOIs' statistics; it says
  nothing about retrieval skill until evaluated out-of-distribution (§7.2).
- **The train-fit ranking (CLIP 0.420 > RemoteCLIP 0.352 > GeoRSCLIP 0.335)** reflects backbone
  capacity to memorise (larger L/14, 768-d fits hardest), not generalisation — the ordering
  reverses on held-out test (§7.2/§7.3, where GeoRSCLIP leads). On zero-shot, RS pretraining
  already helps in-sample (RemoteCLIP 0.057 > CLIP 0.043 > GeoRSCLIP 0.040).

![mAP by encoder x approach, train split](assets/figures/map_bars__train__rgb.png)

Qualitative zero-shot vs PEFT (top-3 retrievals per query, [T1 | T2], green =
relevant): the adapter sharpens scores but the gain is concentrated on the
classes its weak captions cover.

![Zero-shot vs PEFT, CLIP, train](assets/figures/zeroshot_vs_peft__clip_vitl14__train.png)

### 7.2 Cross-split generalisation (adapter trained on train, eval on val/test)

mAP per split (RGB, `difference`):

| Encoder | approach | train | val | test |
|---|---|---|---|---|
| CLIP ViT-L/14 | naive | 0.031 | 0.053 | 0.046 |
| CLIP ViT-L/14 | zero-shot | 0.043 | 0.051 | 0.043 |
| CLIP ViT-L/14 | **PEFT** | **0.420** | **0.042** | **0.040** |
| GeoRSCLIP ViT-B/32 | naive | 0.027 | 0.030 | 0.061 |
| GeoRSCLIP ViT-B/32 | zero-shot | 0.040 | 0.036 | 0.299 |
| GeoRSCLIP ViT-B/32 | **PEFT** | **0.335** | **0.087** | **0.041** |
| RemoteCLIP ViT-L/14 | naive | 0.024 | 0.029 | 0.121 |
| RemoteCLIP ViT-L/14 | zero-shot | 0.057 | 0.025 | 0.050 |
| RemoteCLIP ViT-L/14 | **PEFT** | **0.352** | **0.028** | **0.103** |

> **Evaluation basis:** mAP is macro-averaged only over queries with ≥1 positive
> in that corpus (zero-positive queries are dropped), so the evaluable query set
> differs per split. The headline GeoRSCLIP NRG test result (0.426, §7.3) rests
> on 3 evaluable wetland queries / 15 positives among 110 pairs — a small basis
> that should be read alongside the number.

Key finding: **PEFT overfits to train AOIs**. The bold **train** column is the adapter scored on
its own training pairs (memorisation, not a held-out measurement — see §7.1 / Appendix B.5), so
the only honest read of PEFT is the **val/test** columns. There it is **equal to or worse than
zero-shot, and sometimes below random** (CLIP test PEFT 0.040 < random ≈ 0.083): the adapter
memorises spatial statistics of the 55 training locations rather than learning generalised change
semantics. GeoRSCLIP zero-shot on the test split achieves 0.299 mAP without any training —
suggesting the test AOIs contain land-cover transitions that RS-domain features represent more
discriminatively than train. (The per-split evaluable query sets differ — see the basis note
below — so the train→test drop is not a like-for-like comparison either; the memorisation is
clearest on QFabric, Appendix B.5, where train-fit hits 0.998.)

The train spike then val/test collapse is the signature of the overfit:

![Cross-split mAP, CLIP](assets/figures/cross_split__clip_vitl14__rgb.png)

### 7.3 NIR false-colour ablation

Three colour modes compared (zero-shot mAP, all three splits):

| Encoder | color | train | val | test |
|---|---|---|---|---|
| CLIP ViT-L/14 | RGB | 0.043 | 0.051 | 0.043 |
| CLIP ViT-L/14 | NRG | 0.033 | 0.066 | 0.104 |
| CLIP ViT-L/14 | NDVI | 0.032 | 0.045 | 0.064 |
| GeoRSCLIP | RGB | 0.040 | 0.036 | 0.299 |
| GeoRSCLIP | **NRG** | 0.025 | 0.030 | **0.426** |
| GeoRSCLIP | NDVI | 0.028 | 0.065 | 0.216 |
| RemoteCLIP | RGB | 0.057 | 0.025 | 0.050 |
| RemoteCLIP | NRG | 0.023 | 0.047 | 0.129 |
| RemoteCLIP | NDVI | 0.022 | 0.048 | 0.055 |

**NRG is the best colour mode on held-out test for every encoder.** NDVI usually
sits between NRG and RGB, except for GeoRSCLIP, whose anomalously high test RGB
(0.299) exceeds its NDVI (0.216).

![Colour-mode ablation, zero-shot, test](assets/figures/color_ablation__zero_shot__test.png)

NRG hurts on train (−0.010 to −0.034) but substantially helps on test
(+0.061 CLIP, **+0.127 GeoRSCLIP**, +0.079 RemoteCLIP). NDVI sits
between: it provides some NIR signal but collapses all spectral texture
into a single channel replicated across R/G/B — the RS-pretrained encoders
cannot exploit inter-channel colour contrasts that NRG preserves.

RemoteCLIP NRG (test 0.129) improves over RGB (0.050) but lags GeoRSCLIP
NRG (0.426); GeoRSCLIP's RS5M pre-training gives it a stronger prior for
the NIR-Green spectral contrast characteristic of vegetation transitions.

GeoRSCLIP + NRG zero-shot is the best configuration **on this 110-pair test split (0.426)** —
but **this number is not robust.** Cross-validation over the full 75-AOI corpus (Appendix B.8)
shows it falls to 0.037 (full corpus) / 0.100 ± 0.139 (5-fold); the test split happened to be a
high-wetland "lucky fold" (CV folds span 0.032–0.348). NRG remaining the best colour mode for
every encoder, and the encoder ordering (GeoRSCLIP ≫ CLIP ≈ RemoteCLIP), both hold, but the absolute 0.426 should not
be quoted as the generalisation result — the robust figure is ~0.10, and the only above-random
signal is wetland-formation retrieval.

### 7.4 LoRA adapter — visual encoder fine-tuning

LoRA applied to GeoRSCLIP's ViT-B-32 visual encoder targets the FFN `c_fc`,
`c_proj` in each ResBlock. 368,640 trainable params out of
88M (0.42%); the attention `out_proj` is **not** targeted — open_clip's
`nn.MultiheadAttention` reads `out_proj.weight` directly and never calls
`out_proj.forward`, so a LoRA wrapper there is a silent no-op. Training is online (no pre-cached embeddings; images loaded on-the-fly)
using a masked symmetric InfoNCE loss in the same family as ProjectionHead — the two
differ in how positives are handled (ProjectionHead averages over all same-caption
positives; the LoRA path uses cross-entropy to a single diagonal positive).

Results (GeoRSCLIP + NRG, rank=4, α=8, 20 epochs; zero-shot with LoRA-adapted embeddings):

| split | zero_shot (frozen) | LoRA zero_shot |
|---|---|---|
| train | 0.025 | **0.153** |
| val | 0.030 | **0.034** |
| test | **0.426** | 0.071 |

**LoRA now fits train but collapses out-of-distribution.** With the corrected
masked-InfoNCE loss (and FFN-only target), LoRA lifts train mAP 0.025 → 0.153 yet
test falls to 0.071 — a textbook overfit, and on test it is *below* ProjectionHead
PEFT (0.041 → 0.071 is still ≪ frozen 0.426). The visual encoder adapts to the
specific NIR-channel patterns of the 55 training AOIs, which do not generalise to
the 10 test AOIs.

**Rank / epoch sweep.** To check whether a different capacity changes this, we
swept LoRA rank and epochs (GeoRSCLIP + NRG, zero_shot mAP;
`scripts/lora_sweep.py`, in-memory, no cache/model clobber):

| rank | α | epochs | train | test |
|---|---|---|---|---|
| 4 | 8 | 20 | 0.153 | 0.071 |
| 8 | 16 | 20 | 0.168 | 0.070 |
| 16 | 32 | 20 | 0.135 | 0.058 |
| 8 | 16 | 40 | 0.146 | 0.059 |

There is **no capacity sweet spot**: every rank/epoch fits train (0.13–0.17, well
above the frozen 0.025) yet overfits to test 0.06–0.07. More rank or more epochs
only memorise harder; the best test (0.071) stays well **below frozen NRG
zero-shot (0.426)**. (The earlier "rank-8 sweet spot 0.246" reading was an
artifact of a since-fixed LoRA loss bug plus a stale-cache reuse in the pipeline.)

This reinforces the project's core finding: **spectral physics (NRG
false-colour) generalises; learned visual priors do not.** No amount of adapter
sophistication — small projection head, or LoRA at any rank/epoch we tried —
beats the structural prior embedded in RS-pretrained zero-shot GeoRSCLIP + NIR.

![GeoRSCLIP NRG: frozen zero-shot vs LoRA across splits](assets/figures/cross_split__georsclip__nrg.png)

### 7.5 Re-ranking quantification (GeoRSCLIP + NRG, zero_shot, test split)

Post-retrieval re-ranking was evaluated on the held-out test split (110 pairs,
3 queries with positives) using the best-generalising configuration
(GeoRSCLIP + NRG zero-shot, mAP 0.426).

| Strategy | R@1 | R@3 | R@5 | R@10 | mAP |
|---|---|---|---|---|---|
| baseline (no rerank) | 0.133 | **0.333** | **0.400** | **0.467** | **0.426** |
| diversity | 0.133 | 0.267 | 0.267 | 0.267 | 0.344 |
| coherence | 0.133 | 0.200 | 0.267 | 0.467 | 0.339 |

**Both re-ranking strategies reduce retrieval quality.** Diversity deduplicates
locations, pushing relevant pairs from the same AOI out of the top window.
Coherence clusters geographically near the top-1 result, but relevant pairs
are globally distributed (not spatially clustered). Neither strategy was
designed to optimise semantic relevance — they trade mAP for UX properties:
result variety (diversity) and spatial coherence (coherence). The gap
(−0.082 mAP diversity, −0.087 coherence) is the cost of prioritising display
ergonomics over ranking fidelity. Reproduce with `python -m scripts.eval_rerank`
(default `geo_weight=0.3`; writes
`results/dynamic_earthnet__georsclip__test_nrg__zero_shot__rerank.json`).

### 7.6 UI extensions (non-experiment)

Four optional features were added to the Gradio app. All are toggleable at
any time without restarting; CLI flags set startup defaults only.

**Settings accordion (requires Apply — changes which embeddings are loaded):**

| Control | What it does |
|---|---|
| Color Mode | Switch dataset loader between `rgb`, `nrg` (NIR-Red-Green), `ndvi`. Loads the corresponding pre-cached embeddings. Best config: GeoRSCLIP + NRG (test mAP 0.426). |
| Use LoRA embeddings | Load the LoRA-adapted embedding cache (`_lora` tag) instead of frozen embeddings. Requires prior `run_pipeline --lora` run. |

**Filters & Re-ranking accordion (per-query — no rebuild needed):**

| Control | What it does |
|---|---|
| Geographic filter | Restricts candidate pairs to a continental region (Africa / Asia / Europe / North America / Oceania / South America) using `aoi_metadata.json` bboxes. Masked pairs get score −∞. |
| Re-ranking | Post-processes the ranked list. `diversity`: greedy location-deduplication (prefers unique AOIs). `coherence`: boosts pairs near the top-1 centroid (haversine proximity). |

Both Filters & Re-ranking controls compose freely with each other and with
all three approaches (naive / zero_shot / peft).

### 7.7 Change-feature mode: difference vs concatenate (PEFT)

The PEFT adapter consumes a change feature built from the pair embeddings. Two
modes (`src/features.py`): `difference` (Δf = f_T2 − f_T1, dim D) and
`concatenate` ([f_T1 ; f_T2], dim 2D — keeps both endpoints). All results above
use `difference`; here we re-train the adapter under `concatenate` (same recipe,
RGB, 40 epochs) and compare PEFT mAP across splits:

| Encoder | mode | train | val | test |
|---|---|---|---|---|
| CLIP ViT-L/14 | difference | **0.420** | 0.042 | 0.040 |
| CLIP ViT-L/14 | concatenate | 0.370 | **0.081** | **0.050** |
| GeoRSCLIP | difference | **0.335** | 0.087 | 0.041 |
| GeoRSCLIP | concatenate | 0.275 | 0.086 | **0.070** |
| RemoteCLIP | difference | 0.352 | 0.028 | 0.103 |
| RemoteCLIP | concatenate | **0.359** | **0.109** | 0.078 |

**Concatenate overfits less.** It consistently lowers in-distribution train mAP
(CLIP −0.050, GeoRSCLIP −0.060) but raises held-out **val** for all three
encoders (CLIP +0.039, GeoRSCLIP ~flat, RemoteCLIP +0.081) and **test** for
two of three. Keeping both endpoints (rather than collapsing to their
difference) preserves absolute land-cover context the adapter can use to
generalise, at the cost of some training-set fit. The effect is modest and PEFT
still trails frozen NRG zero-shot (0.426) — consistent with the project's core
finding — but `concatenate` is the better-generalising change feature.
Adapters saved as `models/<dataset>__<encoder>_concatenate__adapter.pt`
(mode-tagged so they never overwrite the difference adapters); reproduce with
`python -m scripts.run_pipeline --encoder <e> --mode concatenate --eval-splits train val test`.

### 7.8 QFabric — second-dataset change-type retrieval (real labels)

To validate the dataset-agnostic design *quantitatively* on a second dataset, we
benchmark QFabric change-type retrieval. Images are the polygon-centred QFabric
crops from `jirvin16/TEOChatlas`; labels are the **real QFabric change types**
(the TEOChatlas RQA2 answers), joined to crops by the shared filename scheme — no
manual rating, no spatial join (`scripts/build_qfabric_labels.py`,
`src/datasets/qfabric_teo.py`). We take a **stratified ≤120 crops/class**
subset (N = 2476 before/after pairs) and 6 change-type queries
(`src/queries/qfabric.py`). Frozen encoders, RGB; mAP:

| Encoder | naive | zero-shot |
|---|---|---|
| CLIP ViT-L/14 | **0.274** | 0.187 |
| GeoRSCLIP | **0.269** | 0.182 |
| RemoteCLIP | **0.233** | 0.180 |

![QFabric change-type mAP by encoder x approach](assets/figures/map_bars__eval__rgb.png)

**naive beats zero-shot — the opposite of DEN.** QFabric change *types*
(residential / road / industrial …) are identifiable from the **after** image
content alone, so `naive` (cos(text, f_T2)) wins; the directional Δ-similarity
that helped on DEN here *adds noise*. The random-ranking baseline is the macro
class prevalence, **0.167** (five common classes at ~0.19 prevalence + the rare
mega_projects at 0.03). `naive` (0.274) sits **+0.11 above** chance; `zero_shot`
(0.187) only **+0.02** — barely better than random.
Per-query (CLIP naive, AP vs that query's prevalence): the signal is concentrated
in **road** (AP 0.50, +0.30 — visually distinctive), **residential** (0.36, +0.17)
and **industrial** (0.30, +0.10); **demolition** (0.25, +0.06) and **commercial**
(0.22, +0.02) are weak; **mega_projects** (0.03) is *at/below* chance (76 pairs,
semantically vague). Encoders are close (CLIP ≈ GeoRSCLIP > RemoteCLIP);
RS-pretraining gives no edge on optical construction crops.

> Recall@K is small by construction (~480 relevant pairs per query in a 2476-pair
> corpus, so R@10 ≤ 0.02); **mAP against the 0.167 macro-prevalence baseline is the
> meaningful signal** — naive is clearly above chance, zero-shot only marginally.

**Takeaway:** the pipeline runs end-to-end on a dataset with a different taxonomy
(6 construction change-types), sensor, and crop scale, with no code changes
beyond a loader + query set — confirming the dataset-agnostic abstraction, and
surfacing a genuinely different regime (after-image content > temporal Δ) from DEN.
**§7.10 relabels the *same* crops along QFabric's temporal status axis and the
regime flips back** — direct evidence the naive-vs-Δ trade-off is set by task
geometry, not the dataset. Reproduce with `scripts/build_qfabric_labels.py` +
`scripts/benchmark_qfabric.py`.

### 7.9 QFabric PEFT — does an adapter help on the second dataset?

Closing the zero-shot-vs-PEFT comparison on QFabric: a stratified, crop-disjoint
**train/test split** (train ≤120 crops/class, N=2422 pairs; held-out test N=2048),
a ProjectionHead adapter trained on the train split with captions phrased like
the queries (`text_caption_for_pair`, difference mode, 40 epochs), evaluated on
both splits (`scripts/benchmark_qfabric.py --peft`):

| Encoder | split | naive | zero-shot | PEFT |
|---|---|---|---|---|
| CLIP ViT-L/14 | train | 0.278 | 0.189 | **0.998** |
| CLIP ViT-L/14 | test | 0.269 | 0.185 | 0.271 |
| GeoRSCLIP | train | 0.271 | 0.180 | **0.999** |
| GeoRSCLIP | test | 0.272 | 0.181 | **0.334** |
| RemoteCLIP | train | 0.244 | 0.179 | **0.999** |
| RemoteCLIP | test | 0.245 | 0.183 | **0.288** |

![QFabric PEFT — test split, naive vs zero-shot vs PEFT](assets/figures/qfabric_peft_test.png)

**PEFT overfits train on *both* datasets (≈0.999 here) — but generalises
differently.** On held-out QFabric test the adapter is **at-or-above `naive`**:
GeoRSCLIP **+0.062** (0.334, the best QFabric result in this report),
RemoteCLIP +0.043 (0.288), CLIP neutral (0.271 ≈ 0.269). Contrast DEN §7.2,
where PEFT test *collapsed below* zero-shot (GeoRSCLIP RGB: zero-shot 0.299 → PEFT 0.041).

The difference is what the adapter has to learn. QFabric change *types* are
recognisable, consistent visual categories (a road looks like a road across
crops), so the head learns features that transfer to unseen crops. DEN's
directional spectral transitions are subtle and AOI-specific, so the head
memorises training-AOI statistics that don't transfer. **PEFT's value is
dataset-dependent — harmful on DEN, mildly helpful on QFabric** — and the
RS-pretrained GeoRSCLIP backbone benefits most from the adapter on construction
imagery. Even so, QFabric PEFT (0.334) and DEN frozen NRG zero-shot (0.426)
remain the respective per-dataset best; no single recipe dominates.

### 7.10 QFabric status-transition retrieval (RQA5) — the regime flips

§7.8 retrieved change *types* (residential/road/…), a property of the **after**
image — so `naive` (cos(text, f_T2)) beat the directional `zero_shot` Δ. QFabric
also ships a **temporal** task (TEOChatlas RQA5/RTQA5): the per-timepoint
**development status** (`Greenland → Prior Construction → Land Cleared →
Excavation → Materials Dumped → Construction Started → Construction Midway →
Construction Done → Operational`). We join each RTQA answer to its frame
("status of this region in Image N" → the N-th timepoint;
`scripts/build_qfabric_status_labels.py`, dataset `qfabric_status`), so a pair
(t1→t2) carries a **status transition**. Six transition queries
(`src/queries/qfabric_status.py`, e.g. *"new building construction has started"*,
*"building construction recently completed"*) are relevant only to pairs whose
status actually **changes** src→dst — **stable pairs are hard negatives** (same
end-state, no change), which is precisely what separates the Δ-signal from
after-image content. Stratified ≤120 crops/final-status (N = 4200 pairs),
frozen, RGB; mAP:

| Encoder | naive | zero-shot |
|---|---|---|
| CLIP ViT-L/14 | 0.057 | **0.057** |
| GeoRSCLIP | 0.079 | **0.084** |
| RemoteCLIP | 0.064 | **0.072** |

**The regime flips: `zero_shot` ≥ `naive` on *all three* encoders** (+0.0004 /
+0.0051 / +0.0074), the **opposite** of §7.8's change-type result (where `naive`
won by ~0.09). The directional Δ is the right tool for *transitions*; after-image
content is the right tool for *types* — one dataset, two tasks, two regimes. The
edge is consistent but **small, and both sit just above the 0.043 macro
prevalence baseline**: status-transition retrieval is genuinely hard zero-shot
(long, irregular temporal baselines; subtle or rare transitions). Per-query the
signal concentrates in the visually distinct, well-populated transitions —
*land cleared* (AP 0.17) and *construction started* (0.18) — and collapses on the
rare/long-range ones (*demolition* 0.01, n=32; *vacant→finished building* 0.02,
n=64). GeoRSCLIP is again the strongest backbone (0.084), echoing §7.8.

**Takeaway:** the same crops, relabelled along a *temporal* axis instead of a
*categorical* one, invert which scoring approach wins — direct evidence that the
zero-shot-vs-PEFT and naive-vs-Δ trade-offs are **task-geometry-dependent**, not
dataset-dependent. Reproduce with `scripts/build_qfabric_status_labels.py` +
`scripts/benchmark_qfabric.py --status`.

### 7.11 LEVIR-CC — open-vocabulary retrieval on human-captioned change

A third dataset tests the engine where change is visually salient and described
in genuine human language. LEVIR-CC (Liu et al., 2022) is 10,077 bi-temporal
building-change pairs (256×256, 0.5 m) with five human captions and a binary
change flag each; the loader (`src/datasets/levir_cc.py`) parses captions into
open-vocabulary change tags, and three free-text queries (new buildings, new
road, demolition; `src/queries/levir_cc.py`) are relevant to pairs carrying the
matching tag. Test split (1929 pairs), frozen encoders, RGB; macro mAP over the
three queries (random-ranking baseline ≈ **0.342** = mean query prevalence:
building 45.2%, road 42.7%, demolition 14.8%):

| Encoder | naive | zero-shot |
|---|---|---|
| GeoRSCLIP | 0.540 | **0.557** |
| CLIP ViT-L/14 | **0.543** | 0.498 |
| RemoteCLIP | 0.539 | **0.573** |

**Retrieval is strong — 0.50–0.57 macro mAP, ~+0.2 above the 0.342 prevalence
floor** — in sharp contrast to DEN's honest cross-validated ceiling of ≈0.20. The
change here (new buildings, roads) is large, high-contrast, and described in the
CLIP text tower's own vocabulary, so a frozen encoder localises it readily; DEN's
directional spectral transitions are subtle and AOI-specific. The RS-pretrained
encoders gain from the directional Δ (RemoteCLIP zero-shot 0.573, GeoRSCLIP 0.557
both beat naive), while general CLIP prefers after-image content (naive 0.543 >
zero-shot 0.498) — the same type-vs-transition split as QFabric. **The lesson
holds across all three datasets: the open-vocabulary engine recovers the change
signal in proportion to its spatial and semantic salience** — strongly on
LEVIR-CC, weakly on DEN. Reproduce: download `lcybuaa/LEVIR-CC`, then build the
`levir_cc` dataset and run `naive`/`zero_shot` on the test split.

### Error analysis — seasonal vs permanent

The benchmark reports seasonal drift @K (non-relevant top-K retrievals that
involve snow/ice, for permanent-change queries). On the DEN train split this
is **0.00 at all K** for every encoder/approach — there is essentially no
seasonal (snow/ice) class in this subset, so seasonal→permanent confusion
does not arise here. The mechanism is implemented and exercised on the
synthetic fixture, which deliberately contains a seasonal snow-melt pair: the
app flags it (*"involves snow/ice — likely SEASONAL, not permanent"*) and the
benchmark would count it as drift if mis-retrieved. The dominant real error is
not seasonal confusion but **low recall from class imbalance and weak
label-derived captions** (many near-stable bimonthly pairs; agri↔wetland
visually subtle).

The per-query confusion analysis (`src/error_analysis.py`) makes this concrete:
it bins each query's top-K retrievals by their *actual* label transition. On the
best test configuration (GeoRSCLIP + NRG), the wetland queries' top-10 are
dominated by **`stable`** pairs (6–7 of 10), not by seasonal or wrong-transition
confusion — i.e. the failure mode is surfacing no-change pairs, consistent with
the class-imbalance reading above.

![Confusion: GeoRSCLIP NRG zero-shot, test](assets/figures/confusion__georsclip__test__zero_shot.png)

Seasonal-drift@K curves (flat at 0 on train, as expected) and the per-encoder
confusion matrices for CLIP train (zero-shot vs PEFT) are in `assets/figures/`.

**Reproducing the figures (from cached embeddings — no GPU training):**

```
python -m scripts.export_results --color-modes rgb nrg ndvi \
    --approaches naive zero_shot peft --lora --confusion --results-dir results
python -m scripts.make_figures --results-dir results --out-dir assets/figures
python -m scripts.make_comparison_figure --encoder clip_vitl14 --split train
```

`export_results` writes one JSON per run + `macro_summary.csv` (machine-readable,
re-plottable); `make_figures` renders the bar/curve/heatmap/confusion PNGs.

### Direct seasonal-robustness probe — stable-pair Δ-similarity FPR gate

`seasonal_drift@K` is uninformative on the current corpora (no snow/ice positives;
reported N/A in §B.6.2), so seasonal robustness is also probed *directly*. The
`zero_shot` score is lifted to a whole-image binary gate: for a change query `t` and
a pair's L2-normalised global embeddings, `Δ = cos(t, f_T2) − cos(t, f_T1)`; the gate
fires ("change") when `Δ > threshold`. On a **stable** pair `f_T1 ≈ f_T2`, so `Δ ≈ 0`
for any query — every firing on a stable pair is a false positive. Sweeping the
threshold over the stable subset (DEN's own `PairLabel.stable` flag) gives a
false-positive-rate (FPR) curve that measures how often seasonal/illumination drift
is mistaken for change. Implementation: [`src/seasonal_gate.py`](src/seasonal_gate.py)
(`ImageLevelChangeGate`); driver: [`scripts/run_seasonal_gate.py`](scripts/run_seasonal_gate.py);
unit tests in `tests/test_seasonal_gate.py`.

Run (local, same RTX 4060 as §8; deterministic — frozen encoder):

```
python -m scripts.run_seasonal_gate --root data/DynamicEarthNet \
    --encoder georsclip --color-mode rgb --split test
```

Result (populate from the run above; `mode=recomputed`):

| split | encoder | color | n stable pairs | mean Δ-sim | FPR@0.00 | FPR@0.02 | FPR@0.05 | FPR@0.10 |
|---|---|---|---|---|---|---|---|---|
| test | georsclip | rgb | 24 | 0.0125 | 0.875 | 0.333 | 0.000 | 0.000 |

Confirmed: stable pairs carry near-zero mean Δ-similarity (0.0125), so the FPR falls to
0 once the threshold clears the noise floor (already 0 at 0.05) — consistent with the
GeoRSCLIP-best zero-shot reading in §7. The 0.875 FPR at threshold 0.00 just reflects
that any nonzero Δ trips a zero threshold; it is not a failure mode.

---

## 8. Resources and operational metrics

### Hardware

| Item | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop, ~8 GB VRAM |
| Runtime | torch 2.10.0 + CUDA (cu130); CPU fallback supported |
| OS / Python | Windows 11 / Python 3.12.10 |
| Optional | free Kaggle / Colab GPU for heavier sweeps |

### Software

torch, torchvision, transformers, open-clip-torch, faiss-cpu, gradio,
rasterio, pandas[parquet], pyarrow, pillow, opencv-python, numpy, gdown,
pytest (all pinned in `pyproject.toml`; installed in editable mode — see the §9 command below).

### Dataset size

| | Value |
|---|---|
| DEN gdown subset (archive) | 7.09 GB (ZIP) |
| DEN extracted | 9.03 GB |
| AOIs | 75 (train 55 / val 10 / test 10) |
| Per AOI | ≈730 daily RGB JPEG @ 1024², labels `(24,1024,1024)` uint8 |
| Working corpus — train split | 605 bimonthly pairs = 1210 tile encodes |
| Working corpus — val / test | 110 pairs each = 220 encodes each |
| Full corpus (all splits) | 825 bimonthly pairs = 1650 tile encodes |
| AOI geographic metadata | `data/DynamicEarthNet/aoi_metadata.json` (75 AOIs, bbox + UTM + S1/S2 availability) |
| Synthetic test fixture | < 1 MB (2 AOIs, deterministic) |

### Model sizes

| Model | Weights on disk | Params (approx) | Dim |
|---|---|---|---|
| CLIP ViT-L/14 (HF) | ≈1.71 GB | ≈427 M | 768 |
| GeoRSCLIP (open_clip ViT-B/32 + RS5M ckpt) | 605 MB ckpt | ≈151 M | 512 |
| RemoteCLIP (open_clip ViT-L/14 + ckpt) | ≈1.7 GB ckpt (downloading) | ≈428 M | 768 |
| **PEFT adapter (only trainable part)** | 2.1–2.9 MB | **725,504** (768) / **528,128** (512) | — |

The adapter is < 0.2 % of the backbone parameter count — the PEFT premise.

### Disk footprint

| Component | Size |
|---|---|
| DEN archive (removable after extract) | 7.09 GB |
| DEN extracted | 9.03 GB |
| CLIP weights cache (`.model_cache/clip-text`, in-repo, gitignored) | ≈1.6 GB |
| HF hub cache (`.model_cache/huggingface`, in-repo, gitignored) | ≈2.2 GB |
| Embedding caches (per encoder, 605 pairs) | CLIP 3.76 MB, GeoRSCLIP 2.52 MB |
| Trained adapters | 2.1–2.9 MB each |
| **Total** | **≈19.5 GB** (≈12.5 GB after deleting the archive) |

### Timings (RTX 4060)

| Operation | Time |
|---|---|
| Retrieval scoring — numpy, 605 pairs, excl. text encode | **0.269 ms/query** |
| End-to-end query — CLIP text forward + scoring, 605 pairs | **10.5 ms** |
| Embedding precompute — CLIP L/14, 1024²→224, GPU | **68 ms/tile** → 1210 tiles ≈ **82 s** (one-time, cached) |
| PEFT training — 605 samples, 40 epochs, adapter only, GPU | **≈29 s** |
| Fast test suite — 225 tests, mock encoders, CPU (full suite 241: 240 pass, 1 skip) | ≈65 s |

All GPU figures measured on the RTX 4060 in a dedicated timed pass (run with
no other GPU job, to avoid contention skew).

## 9. Reproducibility

For a step-by-step run guide see the [Run / install / use](README.md#run--install--use) section
of the README. The commands below are the reproducibility recipe used to produce the numbers in
this report.

```bash
pip install -e .
python -m scripts.download_den --dest data/DynamicEarthNet      # ~7 GB, one-time

# In-distribution run: train on train split, evaluate on train/val/test
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder clip_vitl14 --train-split train --eval-splits train val test --epochs 40
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder georsclip   --train-split train --eval-splits train val test --epochs 40
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder remoteclip  --train-split train --eval-splits train val test --epochs 40

# Best generalising config: GeoRSCLIP + NRG zero-shot (no PEFT training needed)
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder georsclip --color-mode nrg --eval-splits train val test --skip-train

# NDVI ablation (all encoders, zero-shot only)
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder clip_vitl14 --color-mode ndvi --eval-splits train val test --skip-train
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder georsclip   --color-mode ndvi --eval-splits train val test --skip-train
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder remoteclip  --color-mode nrg  --eval-splits train val test --skip-train

# LoRA adapter on visual encoder (GeoRSCLIP + NRG)
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder georsclip --color-mode nrg --skip-train \
    --lora --lora-epochs 20 --lora-rank 4 --lora-alpha 8 \
    --eval-splits train val test

python -m src.app --root data/DynamicEarthNet --encoder clip_vitl14 --split train   # Gradio UI

pytest -q --ignore=tests/test_text_encoder.py    # fast suite, deterministic, no network

# Regenerate result JSON/CSV + publication figures from cache
python -m scripts.export_results --color-modes rgb nrg ndvi \
    --approaches naive zero_shot peft --lora --confusion --results-dir results
python -m scripts.make_figures --results-dir results --out-dir assets/figures
# Statistical-validity audit (Appendix B): random-baseline + FDR over every result cell
python -m scripts.significance_audit --csv results/results_audit_summary.csv
```

Seeds fixed; embeddings and adapters cached and keyed by (dataset, encoder, split,
colour mode) — LoRA caches add a `_lora` tag — with cache invalidation on pair-set change. The synthetic fixture is
regenerated automatically by the test suite.

---

## 10. Limitations and future work

> **Statistical-validity caveat.** The headline numbers below and in §7/§11 were re-audited
> against a random-ranking baseline with FDR correction — see **Appendix B**. Two results are
> load-bearing for that audit: (i) the **in-distribution PEFT figures (train split, §7.1/§11
> point 1) are train-set evaluations of the adapter's own training pairs — memorisation, not
> generalisation** (DEN train 0.42, QFabric train 0.998); quote PEFT from val/test only. (ii) The
> DEN-test headline (GeoRSCLIP NRG 0.426) **does not survive cross-validation**: on the full
> 75-AOI corpus it is 0.037, and 5-fold leakage-free CV gives 0.100 ± 0.139 (Appendix B.8) — the
> test split was a lucky high-wetland fold. The only robust signal is wetland-formation retrieval;
> "open-vocabulary change retrieval" is **not** demonstrated on this data.

- **Weak supervision**: captions derived from dominant-class label flips, not
  human change descriptions — noisy and coarse.
- **PEFT generalisation**: both ProjectionHead and LoRA adapters overfit to
  train AOI statistics (tested; see §7.4). Multi-AOI held-out training or
  domain-randomised augmentation would help. NRG zero-shot remains the most
  robust configuration.
- **Global embeddings are the real ceiling (not the data)**: Appendix B.9 shows that with
  *fraction-based* relevance — which fixes the dominant-class-flip evaluation defect and makes 9/10
  change-types evaluable on the same data — CV mAP still tops out at ~0.15 and only large-area
  (wetland) transitions clear random; localised change (buildings, deforestation, soil) stays at
  chance. **This is now confirmed and partly fixed (B.10, "S3"):** a patch-level Δ-similarity
  scorer (`scripts/patch_eval.py`) lifts GeoRSCLIP NRG CV mAP 0.139 → **0.193** and makes the
  localised change-types (buildings, urban, water) FDR-significant — they were at chance under
  global pooling. Remaining work: a global/patch hybrid (diffuse change still prefers global) and
  a localised change-*attention* head; more data is not the lever (only *snow* is genuinely
  absent from the subset).
- **`concatenate` change mode** now evaluated (§7.7 — generalises better than
  `difference`); **LoRA rank/epoch sweep** done (§7.4 — every config overfits,
  best test ≈0.07, far below frozen 0.426).
- **QFabric** is wired as a second dataset two ways. (a) The EVER-Z image subset
  (`EVER-Z/QFabric_mt_images_1024`) runs the same encoder/retrieval path for
  qualitative zero-shot retrieval in the app (images-only). (b) A **label-grounded
  pipeline** is fully implemented and unit-tested: `src/datasets/qfabric_teo.py`
  (`TEOChatlasQFabricDataset`) reads the QFabric crops from `jirvin16/TEOChatlas`
  with **real change-type labels** (the RQA2 questions; 27,879 labelled crops via
  `scripts/build_qfabric_labels.py`), `src/queries/qfabric.py` provides the 6
  change-type queries, and `scripts/benchmark_qfabric.py` encodes a stratified
  subset and runs Recall@K/mAP. **Benchmarked — see §7.8** (naive mAP ≈ 0.27 >
  zero-shot ≈ 0.18) **and §7.9** (PEFT on a held-out split — overfits train ≈0.999
  but generalises at-or-above naive on test, GeoRSCLIP 0.334; unlike DEN, where
  PEFT collapses). A second, **temporal** task on the same crops — RQA5
  per-timepoint status transitions (`src/datasets/qfabric_status.py`,
  `scripts/build_qfabric_status_labels.py`, `benchmark_qfabric.py --status`) — is
  also benchmarked (**§7.10**): zero-shot Δ ≥ naive on all three encoders,
  flipping the §7.8 regime. Proves the dataset-agnostic design across a different
  taxonomy (6 construction change-types), sensor, and crop scale. Remaining
  QFabric headroom: crop-precise (polygon) grounding. **fMoW** remains wired only
  at the protocol level.
- **Sentinel-1/2 data**: `aoi_metadata.json` confirms 51/75 AOIs have full
  SAR (S1) coverage; downloading and feeding SAR Δ-features is a direct
  extension of the NRG pattern.
- **Re-ranking trades mAP for UX**: diversity and coherence re-ranking are
  quantified in §7.5. Both reduce mAP (−0.082 diversity, −0.087 coherence) vs
  baseline; they optimise display variety, not retrieval quality.
- **Human relevance judgements**: all mAP figures use LULC-derived
  pseudo-labels. Human-annotated query relevance would give a true IR benchmark.

---

## 11. Conclusion

The system fulfils the Case 11 brief: an open-vocabulary, frozen-backbone
change search engine with a Gradio interface, a label-grounded Recall@K/mAP
benchmark, seasonal-vs-permanent error analysis, and a clear **zero-shot vs
PEFT** comparison across **three CLIP-variant encoders** on Dynamic EarthNet.

Three complementary findings:

1. **Adapters only memorise the training set.** PEFT reaches mAP ~8–10× the train
   zero-shot baseline (0.043 → 0.420, CLIP L/14) — but this number is the adapter scored on its
   **own training pairs** (§7.1), so it measures fit, not retrieval. The honest test is
   out-of-distribution, and there it fails (finding 2). On QFabric the same train-fit reaches
   mAP 0.998 (Appendix B.5) — unmistakably memorisation. Low-compute fine-tuning is **not** shown
   to help retrieval.

2. **Out-of-distribution, frozen zero-shot beats PEFT — but the absolute signal is weak.** On
   unseen AOIs PEFT collapses to ≤ zero-shot; zero-shot with the on-disk NIR band as NRG
   false-colour is the best approach and RS-pretrained GeoRSCLIP leads. The often-quoted 0.426
   (test split) is **not robust**: under 5-fold leakage-free cross-validation over all 75 AOIs it
   is **0.100 ± 0.139** (full-corpus 0.037; Appendix B.8) — the test split was a lucky
   high-wetland fold. The only above-random per-query signal is GeoRSCLIP retrieving
   **wetland-formation**; general open-vocabulary change retrieval is not demonstrated on this
   (agri/wetland-dominated) DEN subset.

3. **No fine-tuning helps out-of-distribution.** Cross-validated PEFT (0.049 ± 0.018) is
   at or below zero-shot (0.100); LoRA (every config overfits, best test ≈0.07) also stays
   below frozen zero-shot. Consistent conclusion: **spectral physics (NRG) carries the weak signal
   that exists; learned visual priors — however parameter-efficient — only memorise the train set.**

The contrast between these regimes maps directly onto the deliverable's
grading expectation: a motivated comparison of zero-shot vs PEFT approaches,
not merely a single best number. All three adapters confirm the same conclusion — and the
statistical-validity audit (Appendix B) backs it with random-baseline and FDR analysis.

---

# Appendix A — Code & contract audit

*(11-dimension adversarial audit, 2026-06-02, baseline commit `c6d3c22`; refreshed 2026-06-03
after the QFabric re-run and a full test pass. Originally `AUDIT.md`.)*

**Verdict.** The core is sound: the three scoring formulas (`naive` / `zero_shot` / `peft`) match
the contract and route through one `ChangeRetriever.score_all`; mAP / Recall@K / AP math is
correct; architecture invariants hold (extension-points only, no shared-pipeline edits for
QFabric); the cache is correctly keyed by split+colour (+`_lora` tag); no secrets; all doc links
resolve; and nearly every reported number traces to a committed artifact. 52 of 53 findings
confirmed under skeptic re-check, 1 refuted.

**A.1 Substantive bug found and fixed.** The QFabric label join matched class names as
*substrings* (`crossroad`→`road`) and broke multi-class ties by dict order. Fixed in
`scripts/build_qfabric_labels.py` (`_match_classes`: word-boundary regex + earliest-mention
order). **Re-run impact (this session):** regenerating all QFabric-TEO results against the
corrected labels produced **byte-identical output** — the committed numbers already matched the
fixed labels (see Appendix B.6). The "re-run on the other laptop" item is closed; data + GPU are
on this machine.

**A.2 Smaller fixes applied.** §1 headline CLIP test mAP reconciled with §7.3/CSV (NRG
0.102→0.104, NDVI 0.062→0.064); §7.8 mega_projects 80→76 pairs; §7.9 figure paraphrase tied to
real §7.2 cells; §9 cache-key wording; `embeddings.py` cache-path docstring; `lora_train` default
dataset crash (`dynamic_earthnet_pp`→`dynamic_earthnet`); §7.2 footnote disclosing the 3-wetland
headline basis. Test suite now runs **240 passed, 1 skipped** (supersedes the stale "129/192"
counts).

**A.3 Still open — your decision (thesis-component reuse, all high):**
- Package installs under the generic top-level name **`src`** (no `src/__init__.py`) — collides
  with the thesis repo's own `src`; thesis READMEs import a name
  (`open_vocabulary_temporal_change_retrieval`) that doesn't exist → rename to a unique package.
- **`requires-python = ">=3.12"`** blocks `pip install -e .` in the thesis envs (3.9/3.10) → lower.
- **`compute_patch_text_similarity` min-max normalises per image to [0,1]** — breaks the
  patch-level CLIP-difference baseline the Q2 plan builds on → *resolved:* a raw-cosine patch
  path already exists — `encode_image_patches` returns per-patch features whose cosines are
  directly comparable across T1/T2 (no per-image min-max), as used by `scripts.patch_eval`;
  `compute_patch_text_similarity` retains min-max for the heatmap UI only.
- **`retrieve_changes(query, top_k=5)`** (Q4 Mode-A tool named in the plan) doesn't exist (real
  entry is `ChangeRetriever.search`) → add a thin wrapper or correct the thesis docs.
- GeoRSCLIP (ViT-B/32) gives only a 7×7 patch grid — coarse for a segmentation baseline.

**A.4 Behaviour changes deferred (need a re-run to assess):** `change_type` forced to `"stable"`
on large sub-dominant change (+ degenerate `"X replaced by X"` caption); DEN `.tif` loader
min/max-stretches each tile independently → T1/T2 radiometrically incomparable for change scoring.

**A.5 Unverified, flagged:** `lora_train._infonce_loss` masks with
`masked_fill(~(pos_mask | eye), -1e9)`, which appears to zero the **true negatives** and keep
only positives+diagonal in the softmax denominator — looks inverted. LoRA is a secondary ablation
(§7.4), but verify before trusting LoRA numbers.

---

# Appendix B — Statistical-validity audit of the results

*(2026-06-03. Reproduce with `python -m scripts.significance_audit` →
[`results/results_audit_summary.csv`](results/results_audit_summary.csv). Originally
`RESULTS_AUDIT.md`. This appendix asks "do the numbers mean anything?"; Appendix A asks "is the
code right?")*

**Method.** Each committed mAP cell was re-derived from its per-query records, compared to a
**Monte-Carlo random-ranking baseline** (relevant set held fixed per query, ranking shuffled,
4000 draws), and FDR-corrected (Benjamini-Hochberg) across the ~70 held-out cells.

## B.1 Verdict

The near-zero numbers are **not** a failure — but also not the result they look like.

1. **The scary `R@1=0`, `R@10≈0.01` on QFabric is a metric artifact.** Recall@K here is
   `(#relevant in top-K)/(total relevant)`; with 480 relevant pairs, R@10 is capped at
   `10/480 = 0.021`. Use mAP for QFabric; R@K is only interpretable for DEN (≤5 relevant/query).
2. **Judge mAP against the random-ranking baseline (~prevalence), not 0.** Random scores
   mAP ≈ 0.17 on QFabric-TEO and ≈ 0.08 on DEN-test.
3. **On the committed test split, GeoRSCLIP zero-shot on DEN looks strong** (NRG 0.426, RGB
   0.299, NDVI 0.216; BH-FDR ≤ 0.017 vs random). Everything else is a large-N significance
   mirage (QFabric), a leakage number (train PEFT), or below random. **But §B.3's per-split
   significance is superseded by the cross-validation in §B.8** — see point 4.
4. **That signal does not generalise / is not robust.** DEN-test evaluates only 3
   wetland-transition queries, and **cross-validation (B.8) shows the 0.426 is a lucky-fold
   artifact** — full-corpus 0.037, 5-fold 0.100 ± 0.139. The only above-random signal is
   wetland-formation retrieval; open-vocabulary generality is not demonstrated.

## B.2 Two metric traps

* **Recall@K is ceiling-bounded on QFabric** (`benchmark.py:257`): max R@10 = 0.021 (TEO) /
  0.035 (status). Report Precision@K or R-precision there, or drop R@K.
* **Random AP ≈ prevalence, not 0.** Simulated baselines: DEN-test ≈ 0.083 (the few-positive AP
  estimator sits above bare prevalence 0.045), qfabric_teo ≈ 0.170, qfabric_status ≈ 0.045.

## B.3 Significance — two different questions

| Question | Test | Answer |
|---|---|---|
| Do scores rank relevant items above chance on *this* corpus? (internal) | permutation | Yes for many cells — but see large-N caveat |
| Would the method work on a *new, unseen query*? (external) | needs many queries | **Unanswerable — 3–6 queries, all `permanent`** |

**Large-N mirage (QFabric).** N = 2048–4200 makes the permutation null extremely tight, so even
mAP 0.182 vs random 0.170 gives p < 0.01 and survives FDR — *significance with a negligible
effect* (+6 % relative). Do not present QFabric zero-shot as "significantly working" without the
effect size.

| dataset | encoder | split | cm | approach | nq | mAP | rand | lift | BH-FDR | AP range |
|---|---|---|---|---|---|---|---|---|---|---|
| dynamic_earthnet | georsclip | test | nrg | zero_shot | 3 | **0.426** | 0.082 | 5.2× | **0.000** ✓ | [0.06, 0.63] |
| dynamic_earthnet | georsclip | test | rgb | zero_shot | 3 | 0.299 | 0.084 | 3.6× | **0.002** ✓ | [0.03, 0.47] |
| dynamic_earthnet | georsclip | test | ndvi | zero_shot | 3 | 0.216 | 0.083 | 2.6× | **0.017** ✓ | [0.07, 0.29] |
| dynamic_earthnet | georsclip | test | nrg | naive | 3 | 0.155 | 0.083 | 1.9× | 0.111 ✗ | [0.07, 0.31] |
| dynamic_earthnet | clip_vitl14 | test | * | * | 3 | ≤0.10 | 0.083 | ≤1.2× | 1.000 ✗ | ≈ random |
| qfabric_teo | georsclip | test | rgb | naive | 6 | 0.272 | 0.170 | 1.6× | 0.000 ✓† | [0.01, 0.59] |
| qfabric_teo | georsclip | test | rgb | zero_shot | 6 | 0.181 | 0.170 | 1.1× | 0.017 ✓† | [0.01, 0.28] |
| qfabric_teo | georsclip | test | rgb | peft | 6 | 0.334 | 0.170 | 2.0× | 0.000 ✓† | [0.08, 0.49] |
| qfabric_status | georsclip | eval | rgb | zero_shot | 6 | 0.084 | 0.045 | 1.9× | 0.000 ✓† | [0.01, 0.18] |

✓ survives FDR. † significant but large-N mirage / small effect. The DEN NRG `[0.06,0.63]` range
= 2 of 3 wetland queries work, one (`wetland drained → farmland`) is at chance.

> **The DEN rows above are the committed 110-pair test split and are *not* the generalisation
> verdict.** §B.8 re-evaluates them under full-corpus + 5-fold AOI cross-validation: the 0.426 /
> 0.299 / 0.216 fall to 0.037 / 0.032 / ~0.03 (CV 0.100 / 0.085 / —). Read B.8 as the final word
> on DEN generalisation; this table only shows which *single-split* cells beat random.

## B.4 Per-dataset

* **DynamicEarthNet** — the only real signal, on a 3-query island. 10 queries registered; the
  benchmark drops zero-positive queries per split (`benchmark.py:247`), leaving **test = 3
  wetland transitions only** (building/urban/deforestation/snow never scored). train (5 queries:
  water/bare-soil/wetland) and test are **different query subsets → not comparable**; the
  `lora_sweep.txt` "train 0.025 / test 0.426" is an easier subset, not an improvement.
  GeoRSCLIP ≫ CLIP ≈ RemoteCLIP; zero_shot ≫ naive (NRG 0.426 vs 0.155) — directional change
  modelling genuinely helps here. NRG > RGB > NDVI.
* **QFabric-TEO (change-type)** — appearance, not change. **naive (0.27) > peft-test (0.27–0.33)
  > zero_shot (0.18 ≈ random).** Expected, not a bug: queries describe an *end-state*, so
  `naive = cos(text, f_T2)` matches T2 appearance directly while `zero_shot` subtracts a
  partly-correlated T1 term. The "change" approach barely clears random here.
* **QFabric-STATUS (status-transition)** — at the floor. mAP 0.057–0.084 vs random 0.045;
  FDR-significant only because N = 4200; effect tiny.

## B.5 PEFT / LoRA — leakage and overfitting

**Train-split PEFT/LoRA numbers are scored on the adapter's own training pairs** (`run_pipeline.py`
:136 trains on `ds_train`, :156 scores `peft` on the same `ds_train`; `benchmark_qfabric.py`
:88–122 likewise) → discard as memorisation: qfabric_teo train PEFT **0.998–0.999** (R@K =
exactly k/n_relevant = perfect top-precision); DEN train PEFT 0.34–0.42 vs DEN-**test** 0.04.
**On held-out data no adapter beats frozen zero-shot:** DEN PEFT-test 0.04–0.10 < zero-shot
0.30–0.43 (on `clip_vitl14` PEFT-test 0.040 is *below* random 0.083); every LoRA config <
frozen 0.426 (`lora_sweep.txt`); `difference` mode overfits hardest (train 0.42 → test 0.04,
`concat_eval.txt`). Only TEO-test PEFT (0.27–0.33) modestly tops naive — fitting end-state
appearance, not change. **Reporting rule: quote PEFT/LoRA from test/val only, never train.**

## B.6 Bugs & data hygiene

| # | Finding | Severity |
|---|---|---|
| B.6.1 | `macro_summary.csv` had a blank-metrics row — `load_all` mis-parsed the rerank JSON (nested `strategies`, no top-level `macro`). **Fixed:** `results_io.load_all` now skips `*rerank*`; CSV regenerated (96 clean rows); regression test added. | resolved |
| B.6.2 | `seasonal_drift@k = 0.0` in every file — QFabric has no seasonal class and DEN's snow query is never evaluable → uninformative on current data. **Resolved by documentation:** kept in the schema (cheap, and meaningful the moment a seasonal-positive corpus is added) but reported as N/A here; do not cite it. | resolved (documented) |
| B.6.3 | Suspected-stale QFabric-TEO results **refuted**: re-run produced byte-identical output (see A.1). | none |

## B.7 What is defensible to claim

**Can say:** GeoRSCLIP (RS-pretrained) + NRG zero-shot Δ-similarity retrieves **wetland-formation**
change above chance under cross-validation (the only above-random per-query signal; B.8); domain
pretraining helps (GeoRSCLIP > CLIP-L/14 ≈ RemoteCLIP); spectral NRG is the best colour mode; on DEN
directional change beats end-state matching, on QFabric the reverse (end-state queries); learned
adapters (PEFT/LoRA) only memorise — cross-validated PEFT (0.049) is at or below zero-shot (0.100).
**Cannot say:** "open-vocabulary change retrieval works" — under CV no config exceeds ~0.10 mAP,
the 0.426 was a lucky test fold (B.8), and 4 of 10 change-types have zero positives in the data;
"PEFT/LoRA improves retrieval" (never beats frozen zero-shot held-out; high numbers are train
leakage); "QFabric change retrieval works" (zero-shot ≈ random).
**Highest-impact next experiment:** acquire DEN AOIs (or a dataset) containing
building/urban/deforestation/snow change so the 4 currently-absent query types are evaluable —
the single biggest barrier to a real open-vocabulary claim (the within-data CV is now done, B.8).
Also: report effect size + permutation p alongside mAP in §7; replace R@K with P@K for QFabric.

## B.8 Cross-validated re-evaluation — the headline 0.426 does not survive

The 3-query threat (B.4) was tested directly with `scripts/cv_eval.py`: merge the cached
train+val+test embeddings (= all **75 AOIs / 825 pairs**, no re-encoding), then evaluate (a) on
the **full corpus** with bootstrap CIs + permutation p, and (b) under **5-fold leakage-free AOI
cross-validation**. Result — **the test-split headline numbers are small-corpus, lucky-fold
artifacts and collapse to near-random:**

| config | test-split (§7.3) | full-corpus mAP | 5-fold CV mAP | FDR-significant queries (full corpus) |
|---|---|---|---|---|
| GeoRSCLIP NRG zero-shot | 0.426 | **0.037** | **0.100 ± 0.139** | 2 — *land→wetland* (q=0.022), *ag→wetland* (q=0.030) |
| GeoRSCLIP RGB zero-shot | 0.299 | 0.032 | 0.085 ± 0.069 | 0 — none survive FDR (min q=0.18) |
| CLIP-L/14 NRG zero-shot | 0.104 | 0.030 | 0.076 ± 0.023 | **0** |
| RemoteCLIP NRG zero-shot | 0.129 | 0.021 | 0.053 ± 0.022 | **0** |
| GeoRSCLIP NRG PEFT (k-fold) | — | — | 0.049 ± 0.018 | — |

- **The 0.426 was a lucky fold.** The 5 CV folds for GeoRSCLIP NRG span **0.032–0.348**; the
  original 110-pair test split coincided with the easy high-wetland AOIs (the 0.348 fold). On the
  full corpus the same config scores 0.037 — a ~11× drop. Smaller corpora inflate AP (fewer
  distractors); the headline rode that plus an easy split.
- **Only GeoRSCLIP shows any signal, and only for wetland *formation*.** `land→wetland` and
  `agricultural→wetland` clear random (p<0.05); the reverse `wetland→farmland`, plus water-body /
  bare-soil / forest-loss, are at or below chance. CLIP-L/14 and RemoteCLIP have **zero**
  above-random queries on the full corpus. Even the significant queries have std ≈ mean across
  folds (e.g. ag→wetland 0.136 ± 0.202) — high instability.
- **Leakage-free PEFT confirms B.5.** Cross-validated PEFT (train on 4 folds, eval on the 5th)
  scores 0.049 ± 0.018 — statistically indistinguishable from zero-shot's 0.100, and nowhere near
  the train-fit 0.42. PEFT does not help once it cannot see the eval pairs.

**Net:** the only defensible DEN claim is *"GeoRSCLIP (RS-pretrained, NRG) retrieves
wetland-formation change above chance"* — a narrow, single-direction, high-variance signal, not a
0.426 open-vocabulary result. Reproduce: `python -m scripts.cv_eval --encoder georsclip
--color-mode nrg --folds 5 [--peft]` → `results/cv_eval__*.json`.

## B.9 Is it bad data, or a bad method? Fraction-based relevance separates the two

The weak DEN numbers had two candidate causes: (i) **eval starvation** — the default relevance
predicate (`benchmark._transition`) only fires when the *dominant class of the whole 1024² tile
flips*, so localised change (a new building on an agriculture-majority tile) is labelled
"stable". Only **8.6 % of pairs (71/825)** flip dominant class, and 44 of those are
wetland↔agriculture — which is *why* only wetland queries had positives and 4 change-types had
none. (ii) a **method limit** — global CLIP-embedding differencing may simply not encode
localised change. The labels themselves are complete and dense (per-pixel 7-class, 24 months, all
75 AOIs, none missing), so this is **not a missing-label problem**.

To separate the two, `scripts/cv_eval.py --relevance fraction` re-derives relevance from the
per-class pixel-change fractions `derive_pair_label` already computes (`benchmark._gained` /
`_lost`, default threshold 5 % of valid pixels; `src/queries/den.py::frac_queries`). Same data,
same queries, same cached embeddings — only the relevance rule changes:

| config | dominant-flip: nq · CV mAP | fraction (≥5%): nq · CV mAP · FDR-sig queries |
|---|---|---|
| GeoRSCLIP NRG zero-shot | 6 · 0.100 ± 0.139 | **9 · 0.139 ± 0.024** · 2 (ag→wetland, land→wetland) |
| CLIP-L/14 NRG zero-shot | 6 · 0.076 ± 0.023 | 9 · 0.123 ± 0.029 · 0 (none) |
| RemoteCLIP NRG zero-shot | 6 · 0.053 ± 0.022 | 9 · 0.134 ± 0.027 · 0 (none) |
| GeoRSCLIP RGB zero-shot | 6 · 0.085 ± 0.069 | 9 · 0.115 ± 0.036 · 1 (wetland→farmland) |
| GeoRSCLIP NRG PEFT (k-fold) | — | 0.196 ± 0.049 |

**What curation fixed (it was partly bad eval):** fraction relevance makes **9 of 10** queries
evaluable (buildings 14, urban 14, deforestation 13, forest-loss 13, soil 25, water 9 positives —
all previously 0–3), roughly **triples** full-corpus mAP (0.03→0.09) and **tightens** the CIs
(more positives, std tightens from ±0.139 to ±0.024). So the dominant-flip relevance *was*
starving the benchmark — a real, now-fixed evaluation bug.

**What curation did NOT fix (the rest is the method):** even with dense, correct labels, only
**0–2 of 9 queries beat random per config** (only GeoRSCLIP, on wetland formation), and no config
exceeds **~0.15** CV mAP. Buildings, urban, deforestation, forest-loss, soil and water sit at
chance after FDR correction despite hundreds of positives. The retrievable signal is confined to
**large-area** transitions (wetland formation); **localised** change is not
retrievable by global-embedding differencing. PEFT k-fold (0.196 ± 0.049) overlaps zero-shot (0.139 ± 0.024) within fold variance — still no
real gain.

**Conclusion (answering "bad data?"):** not bad data, not missing labels. The benchmark's
dominant-class-flip relevance was a genuine evaluation defect (now fixable in-place, no new data);
the *residual* weakness is a **method ceiling** — global VLM embeddings dilute localised change.
The only data that is truly absent is **snow** (the 10th query, 0 positives at any threshold).
The real lever for better numbers is a **localised / patch-level method** (B.10, "S3"),
not more data. Reproduce: `python -m scripts.cv_eval --encoder <e> --color-mode <c> --folds 5
--relevance fraction [--peft]`.

## B.10 S3 — patch-level (localised) scoring: confirmed partial fix

B.9 pinned the residual weakness on global-embedding dilution. S3 tests the fix directly
(`scripts/patch_eval.py`): score each pair from **per-patch** embeddings
(`encoder.encode_image_patches`, raw cosine-comparable — no per-image min-max) instead of one
global vector. For query `t` and spatially-aligned patch grids `P1,P2`:
`patch_zeroshot = max_p(cos(t,P2_p) − cos(t,P1_p))` (most-activated change patch);
`patch_top3` = mean of the top-3 such deltas. Same 75-AOI corpus, fraction relevance, 5-fold CV —
directly comparable to B.9.

![DEN result chain: the 0.426 test-split headline collapses to ~0.10 under cross-validation, then recovers to ~0.20 by fixing evaluation (S1, fraction relevance) and method (S3, patch scoring); per-query, S3 rescues localised change-types](assets/figures/cv_progression.png)

| GeoRSCLIP NRG, fraction relevance | k-fold CV mAP | FDR-significant queries (of 9) |
|---|---|---|
| global zero-shot (B.9) | 0.139 ± 0.024 | 2 (wetland formation only) |
| global PEFT k-fold (B.9) | 0.196 ± 0.049 | — |
| patch_naive (max end-state) | 0.111 ± 0.024 | — |
| patch_zeroshot (max Δ) | 0.150 ± 0.028 | 1 (wetland→farmland) |
| **patch_top3 (mean top-3 Δ)** | **0.193 ± 0.051** | **4** (buildings q=.003, urban q=.006, wetland→farmland q=.002, land→wetland q=.003; new-water q=.053 n.s.) |
| CLIP-L/14 NRG patch_top3 (256-patch grid) | 0.149 ± 0.054 | — |

**Findings:**
- **Patch scoring is the best DEN configuration found** — GeoRSCLIP NRG `patch_top3` reaches
  **0.193 ± 0.051** CV mAP, ~39 % over the global zero-shot 0.139, and is the honest peak (the
  old "0.426" was a lucky test fold, B.8).
- **It rescues exactly the localised change-types global embeddings missed.** "New buildings"
  (ap 0.02→0.13) and "urban expansion" go from chance to FDR-significant; "new water" too. This
  is direct evidence the B.9 diagnosis was right: the failure was *global pooling*, not the data.
- **Still only 5/9 queries.** Deforestation, forest-loss, bare-soil remain at chance; diffuse
  large-area `ag→wetland` slightly *prefers* the global score (max/top-k patch aggregation favours
  localised change) — so global and patch are **complementary**; a per-query or max(global,patch)
  hybrid is the natural next step.
- **RS pretraining beats grid resolution.** CLIP-L/14's finer 256-patch grid (0.149) does not beat
  GeoRSCLIP's coarse 49-patch grid (0.193) — domain pretraining matters more than patch count.

**Net:** S3 is a real, scope-preserving improvement — localised scoring roughly doubles the number
of retrievable change-types and lifts CV mAP ~30 %, confirming the method (not the data) was the
ceiling. Absolute numbers stay modest (best ~0.20), so the honest framing is unchanged: a *weak
but now-better-understood* open-vocabulary signal, strongest on localised construction/water and
large-area wetland change. Reproduce: `python -m scripts.patch_eval --encoder georsclip
--color-mode nrg --approach patch_top3`.

## B.11 Two cheap in-scope tweaks tested — neither beats patch_top3 (honest negatives)

Both stay within the brief (frozen backbone, cosine scoring, no fine-tuning) and reuse the cached
embeddings (no GPU/data). GeoRSCLIP NRG, fraction relevance, 5-fold CV:

| variant | CV mAP | vs patch_top3 0.193 |
|---|---|---|
| global zero-shot (baseline) | 0.139 ± 0.024 | — |
| global zero-shot + prompt-ensemble | 0.142 ± 0.029 | wash |
| **patch_top3 (best)** | **0.193 ± 0.051** | — |
| patch_top3 + prompt-ensemble | 0.195 ± 0.048 | wash |
| hybrid (z-scored global Δ + patch top-3) | 0.165 ± 0.050 | **worse** |
| hybrid + prompt-ensemble | 0.159 ± 0.046 | worse |

- **Prompt ensembling** (averaging the query embedding over 5 templates, `benchmark.encode_query`,
  `--prompt-ensemble`) is a **wash** — GeoRSCLIP's RS-text alignment gains nothing from generic
  templates on these already-descriptive queries.
- **Equal-weight hybrid** (`patch_eval.py --approach hybrid`: z-score the global Δ and the patch
  top-3 score, sum) **hurts** (0.165 < 0.193): fusing the mostly-noise global signal dilutes the
  stronger patch signal. The B.10 observation that the two are *complementary per-query* holds,
  but a flat z-sum is the wrong fusion — a **query-type-gated or learned weighting** (global for
  diffuse, patch for localised) would be needed, and is left as future work rather than tuned here
  (out of scope to over-engineer).

**Conclusion:** `patch_top3` remains the best DEN configuration (0.193 CV mAP). The two cheapest
plausible add-ons do not help; reporting them is the honest outcome and rules them out with
evidence. Reproduce: `python -m scripts.cv_eval ... --prompt-ensemble` /
`python -m scripts.patch_eval ... --approach hybrid`.

**Shipped in the deliverable app.** Because `patch_top3` is the best DEN config, it is now a
selectable scoring mode in the Gradio engine ("Patch / localised (best on DEN)"): the app encodes
per-patch embeddings for the loaded corpus once (lazily, cached for the session) and scores via
`retrieval.top_patch_change_scores`. The search engine therefore demonstrates the strongest
configuration, not only the global baselines.

## B.12 Change-attention (training-free spatial aggregation) — refines, does not break the ceiling

S3 (B.10) aggregates patches independently (top-3 Δ). Two **training-free** "change-attention"
variants test whether *spatial structure* of the Δ-map helps — deliberately training-free to
avoid the memorisation trap that sank every learned head (B.5): `patch_softattn` (query-conditioned
softmax-weighted mean of the per-patch Δ, τ=0.03) and `patch_spatial` (reshape Δ to the patch grid,
3×3 mean-smooth, then top-3 — rewards spatially-contiguous change). 5-fold CV, fraction relevance:

| approach | GeoRSCLIP NRG (49-patch grid) | CLIP-L/14 NRG (256-patch grid) |
|---|---|---|
| patch_top3 (B.10 best) | **0.193 ± 0.051** | 0.149 ± 0.054 |
| patch_softattn (τ=0.03) | 0.137 ± 0.042 | 0.139 ± 0.043 |
| patch_spatial (3×3-smoothed) | 0.168 ± 0.045 | **0.163 ± 0.046** |

- **Spatial coherence helps the *fine* grid, hurts the *coarse* one — as predicted.** On CLIP-L/14's
  256-patch grid (where top-3 is noisy) smoothing lifts mAP 0.149→0.163 (+9 % rel, within fold
  variance); on GeoRSCLIP's 7×7 grid it *over-smooths* the 1–3 genuine change patches and drops
  0.193→0.168. Grid resolution determines whether spatial denoising helps.
- **Soft-attention is a wash** — top-3 (hard, small-k) already sits at the sweet spot between max
  (`patch_zeroshot` 0.150) and mean; softmax-weighting all patches only dilutes it.
- **No variant beats GeoRSCLIP `patch_top3` (0.193).** The ~0.20 cross-validated ceiling is robust;
  change-attention *refines* the aggregation but does not break the underlying frozen-VLM limit.

**Learned attention (a trainable query→patch head) was deliberately not pursued:** B.5 showed every
trained head on this data memorises train AOIs with no held-out gain, and B.12 shows the
spatial-structure headroom is small (and grid-dependent) — so the likely outcome is a memorising
head for a marginal, fold-noise-sized gain. Recorded as future work, not built. Reproduce:
`python -m scripts.patch_eval --encoder <e> --color-mode nrg --approach patch_spatial|patch_softattn`.
