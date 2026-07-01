---
library_name: peft
tags:
- lora
- remote-sensing
- change-detection
- clip
base_model: Zilun/GeoRSCLIP
license: mit
---

# LoRA adapter — GeoRSCLIP visual encoder for open-vocabulary temporal change retrieval

A LoRA adapter on the **GeoRSCLIP** visual encoder for bi-temporal, open-vocabulary change
retrieval on **Dynamic EarthNet** in NRG (near-infrared / red / green) colour mode. Part of the
GBDA lab project — full code and methodology in the
[repository](https://github.com/Panagiotis427/MSc_GBDA-OV_Temporal_Change_Retrieval) and its
[technical report](https://github.com/Panagiotis427/MSc_GBDA-OV_Temporal_Change_Retrieval/blob/main/report/main.pdf).

## Honest summary — a negative result

**This adapter is a research artifact, not a recommended configuration.** Under leakage-free
5-fold leave-AOI-out cross-validation it **memorises the training AOIs** (high train mAP) but
gives **no held-out gain** over the frozen zero-shot encoder. For real retrieval, use the frozen
**GeoRSCLIP + NRG** encoder with zero-shot / patch-level Δ-scoring — the report's best
configuration (CV mAP 0.193 ± 0.051). This card documents the comparison for reproducibility, not
for deployment.

## Training

- **Base model:** GeoRSCLIP (RS5M-pretrained CLIP, ViT-B/32, 512-d). Only the visual encoder is
  LoRA-adapted; the rest of the backbone stays frozen.
- **Data:** Dynamic EarthNet, `train` split, NRG colour mode; weak "X replaced by Y" change
  captions per bi-temporal pair.
- **Objective:** contrastive (masked symmetric InfoNCE) over bi-temporal change features.
- **Config / procedure:** see `src/lora_train.py` and `scripts/run_pipeline.py --lora` in the
  repository for the exact LoRA rank / alpha / epochs and training loop.

## Usage

Not a standalone model — load it through the repository:
`python -m scripts.run_pipeline … --lora`, or toggle **LoRA** in the app's Settings panel. See the
[repository README](https://github.com/Panagiotis427/MSc_GBDA-OV_Temporal_Change_Retrieval).

## License

MIT — see the repository's `LICENSE`.

### Framework versions
- PEFT 0.18.1
