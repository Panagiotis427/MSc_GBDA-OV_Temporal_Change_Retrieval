# Quickstart — Semantic Change Search Engine

Type a natural-language land-cover change query; get ranked satellite image-pair
matches with a heatmap and confidence score.

**Just running the app?** Follow sections 1–3.  
**Hosted demo (no install)?** Push the repo to a HuggingFace Space — `app.py` and `requirements.txt` are ready. See [`docs/EXTENSIONS.md`](docs/EXTENSIONS.md).  
**Retraining or extending?** See section 4 and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 1. Setup (one-time)

**Requirements:** Python 3.12+ · ~3 GB disk (model weights) · ~9 GB more for real DEN · GPU optional

```bash
git clone <repo-url> && cd MSc_GBDA-OV_Temporal_Change_Retrieval

python -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
                                   #   if blocked once: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

pip install -e .
```

---

## 2. Option A — 30-second synthetic demo (no download)

```bash
python -m scripts.make_den_fixture
# Builds tests/fixtures/den_tiny/: 2 AOIs × 8 months, <1 MB, deterministic.

python -m src.app --root tests/fixtures/den_tiny --split all --encoder clip_vitl14
# First run downloads CLIP weights (~1.6 GB) into .model_cache/ — one-time.
# Open http://127.0.0.1:7860
```

---

## 3. Option B — real Dynamic EarthNet (~7 GB)

```bash
python -m scripts.download_den --dest data/DynamicEarthNet
# ~7 GB ZIP via gdown; extracted; idempotent (_done.marker guards re-runs).

python -m src.app --root data/DynamicEarthNet --encoder clip_vitl14
# Defaults: --split train (55 AOIs, 605 pairs), --approach zero_shot.
# Switch to --approach peft in the UI for the trained-adapter scoring.
# Open http://127.0.0.1:7860
```

---

## App usage

Enter a query, press **Search**. Example queries:

- `agricultural land converted to wetland`
- `new buildings on former farmland`
- `forest cleared to bare soil`

Results: T1 / T2 tiles side by side · heatmap on T2 · confidence (0–1) ·
permanence note (`permanent` / `likely SEASONAL` / `stable`) · ranked table.

### Launch-time flags

Two accordions hold the controls:

- **Settings** (requires **Apply** to rebuild embeddings): Dataset, Encoder, Approach, Color Mode, LoRA.
- **Filters & Re-ranking** (takes effect on next **Search**, no Apply): Geographic filter, Re-ranking.

All options can also be set as startup defaults via CLI flags:

| Flag | Default | Notes |
|---|---|---|
| `--root` | `data/DynamicEarthNet` | Path to dataset; DEN layout auto-detected. |
| `--split` | `train` | DEN AOI split: `train` (605 pairs), `val`/`test` (110 each), `all` (825). |
| `--pairing` | `bimonthly` | How DEN's 24 monthly timesteps pair into (T1, T2). |
| `--port` | `7860` | Gradio HTTP port. |
| `--color-mode` | `rgb` | `rgb` / `nrg` (NIR-Red-Green, best zero-shot with GeoRSCLIP) / `ndvi`. Toggle in Settings. |
| `--lora` / `--no-lora` | off | Load LoRA-adapted embeddings (must be pre-cached by `run_pipeline --lora`). Toggle in Settings. |
| `--geo-filter` / `--no-geo-filter` | off | Enable geographic region filter at startup. Toggle in Filters & Re-ranking. |
| `--rerank` / `--no-rerank` | off | Enable post-retrieval re-ranking at startup. Toggle in Filters & Re-ranking. |
| `--rerank-strategy` | `diversity` | `diversity` = unique AOIs per result; `coherence` = cluster near top-1 location. |

### Troubleshooting

| Symptom | Fix |
|---|---|
| Port already in use | Add `--port 7861`; visit `http://127.0.0.1:7861`. |
| First launch slow | CLIP weights download once (~1.6 GB) into `.model_cache/`. |
| `peft` errors "no adapter" | Adapter file missing from `models/`; train with `run_pipeline` (section 4) or switch to `zero_shot`. |
| venv not activating | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once in PowerShell. |

---

## 4. Developer — pipeline, training, tests

### Full pipeline (embed → benchmark → PEFT → cross-split table)

```bash
# Train on train split, evaluate on all three splits:
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder clip_vitl14 --train-split train --eval-splits train val test --epochs 40

# Best zero-shot generalisation (GeoRSCLIP + NIR, no training):
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder georsclip --color-mode nrg --eval-splits train val test --skip-train

# LoRA adapter on visual encoder (add alongside or instead of PEFT):
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder georsclip --color-mode nrg --skip-train \
    --lora --lora-epochs 20 --lora-rank 4 --lora-alpha 8 \
    --eval-splits train val test
```

Repeat with `--encoder georsclip` / `--encoder remoteclip` for the three-encoder
comparison in `REPORT.md §7`.

### Individual stages

> These are convenience entry points; the canonical, cache-consistent flow is
> `scripts.run_pipeline` (above). Pass the same `--split` / `--color-mode` to every
> stage so they share the split-tagged embedding cache.

```bash
python -m src.embeddings --root data/DynamicEarthNet --encoder clip_vitl14 \
    --split train --color-mode rgb
# Precompute embeddings only. Cache: data/cache/<dataset>__<encoder>__train__pair_embeddings.npz

python -m src.benchmark  --root data/DynamicEarthNet --encoder clip_vitl14 --approach all
# Recall@K / mAP / seasonal drift on cached embeddings.

python -m src.train      --root data/DynamicEarthNet --encoder clip_vitl14 --split train
# Train PEFT adapter on the train split → models/<dataset>__<encoder>__adapter.pt
# (omit --split and it defaults to the 110-pair 'test' split — not what the report trains on)

python -m src.lora_train --root data/DynamicEarthNet --encoder georsclip --color-mode nrg
# Train LoRA adapter on visual encoder; merges + re-caches embeddings automatically.
```

### Tests

```bash
pytest -q --ignore=tests/test_text_encoder.py
# 192 tests, ~20 s on CPU. Mock encoders; no network; covers full pipeline.

pytest tests/test_text_encoder.py
# Real CLIP weights; ~45 s. Verifies text-encoder fix end-to-end.
```

### Adding datasets / encoders

File-additive only — never edit shared pipeline files. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### Reference docs

- [`README.md`](README.md) — pipeline diagram, module map.
- [`REPORT.md`](REPORT.md) — all runs, metrics, timings, error analysis.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — extension contract.
- [`docs/Common_Resources.md`](docs/Common_Resources.md) — dataset / model links.
