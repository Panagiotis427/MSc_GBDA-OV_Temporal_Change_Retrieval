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
