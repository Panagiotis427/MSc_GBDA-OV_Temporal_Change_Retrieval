"""
RemoteCLIP encoder — the first VLM purpose-built for remote sensing
(Liu et al., IEEE TGRS).

Weights ship as a plain checkpoint in the ``chendelong/RemoteCLIP``
HuggingFace repo, loaded onto a standard open_clip ViT-L-14 (768-d shared
space — matches CLIP ViT-L/14, so adapters/indexes are dimension-compatible).
Implements the project's ``ImageTextEncoder`` protocol via
:class:`OpenClipHFEncoder`.
"""
from __future__ import annotations

from ._openclip_base import OpenClipHFEncoder


class RemoteCLIPEncoder(OpenClipHFEncoder):
    name = "remoteclip"
    embed_dim = 768
    image_input_size = 224
    _arch = "ViT-L-14"
    _hf_repo = "chendelong/RemoteCLIP"
    _hf_file = "RemoteCLIP-ViT-L-14.pt"
