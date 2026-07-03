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
    # Pin the checkpoint: commit revision (immutable HF tree) + content SHA-256
    # (git-LFS oid = file digest), so a swapped upstream file is refused before
    # the weights_only=False load. See OpenClipHFEncoder._load_state_dict_flexible.
    _hf_revision = "bf1d8a3ccf2ddbf7c875705e46373bfe542bce38"
    _hf_sha256 = "fcc2a7e21e171f4ffcb7a9c0206b8b74ac0c9eb83c67b576958b7a4ed6c8cecb"
