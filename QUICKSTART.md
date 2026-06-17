# Quickstart

Open-vocabulary bi-temporal change **retrieval**. A natural-language query is
scored against every image pair's *change* representation; results are a ranked
list of change events with a query-conditioned change heatmap. See
[`README.md`](README.md) for the full tour and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
for the design contract.

Run everything from the repo root.

```bash
pip install -e .
```

## 1. 30-second synthetic demo (no download)

```bash
python -m scripts.make_den_fixture
python -m src.app --dataset dynamic_earthnet --root tests/fixtures/den_tiny --split all --encoder clip_vitl14
# Gradio UI at http://127.0.0.1:7860
```

## 2. Real Dynamic EarthNet (~7 GB via gdown)

```bash
python -m scripts.download_den --dest data/DynamicEarthNet
python -m src.app --dataset dynamic_earthnet --root data/DynamicEarthNet --encoder clip_vitl14
```

In the UI, **Approach** switches between `naive`, `zero_shot`, `patch` (localised),
and `peft` (needs a trained adapter — see below). The app binds to `127.0.0.1` by
default; pass `--host 0.0.0.0` to expose it on the LAN.

## 3. Full pipeline: embed → benchmark → PEFT → cross-split table

```bash
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder clip_vitl14 --train-split train --eval-splits train val test --epochs 40
```

`run_pipeline` is the canonical, cache-consistent flow. It is idempotent: a valid
embedding cache is reused (logged `mode=reused`); pass `--force` to recompute, or
`--dry-run` to print the planned steps without computing. Best zero-shot
generalisation is `--encoder georsclip --color-mode nrg --skip-train`.

## 4. Tests

```bash
pytest -q --ignore=tests/test_text_encoder.py   # fast suite (no network)
pytest tests/test_text_encoder.py               # requires real CLIP weights (~45 s)
```
