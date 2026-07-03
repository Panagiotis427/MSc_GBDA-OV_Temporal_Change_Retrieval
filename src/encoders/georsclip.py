"""
GeoRSCLIP encoder — CLIP fine-tuned on RS5M (Om AI Lab).

Weights ship as a plain checkpoint in the ``Zilun/GeoRSCLIP`` HuggingFace
repo, loaded onto a standard open_clip ViT-B-32 (512-d shared space).
Implements the project's ``ImageTextEncoder`` protocol via
:class:`OpenClipHFEncoder`.
"""
from __future__ import annotations

from ._openclip_base import OpenClipHFEncoder


class GeoRSCLIPEncoder(OpenClipHFEncoder):
    name = "georsclip"
    embed_dim = 512
    image_input_size = 224
    _arch = "ViT-B-32"
    _hf_repo = "Zilun/GeoRSCLIP"
    _hf_file = "ckpt/RS5M_ViT-B-32.pt"
    # Pin the checkpoint: commit revision (immutable HF tree) + content SHA-256
    # (git-LFS oid = file digest), so a swapped upstream file is refused before
    # the weights_only=False load. See OpenClipHFEncoder._load_state_dict_flexible.
    _hf_revision = "4920188e6eba4e711ef9848cfd7cb77e874ee33f"
    _hf_sha256 = "129bafaa6a097b8be52e2babf27d24f0a934dae919201e538dc698611bd1ea01"
