"""
Single in-repo cache root for all downloaded model weights, so nothing is
dispersed to ``~/.cache``. Import this module *before* transformers /
huggingface_hub / open_clip so ``HF_HOME`` is set in time.
"""
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
MODEL_CACHE = _REPO / ".model_cache"
HF_HOME = MODEL_CACHE / "huggingface"
CLIP_CACHE = MODEL_CACHE / "clip-text"

MODEL_CACHE.mkdir(parents=True, exist_ok=True)
# Only set if the user hasn't overridden, so external HF_HOME still wins.
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))

CLIP_CACHE_DIR = str(CLIP_CACHE)
