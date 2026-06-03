"""
Shared base for ``open_clip``-architecture encoders whose weights ship as a
plain checkpoint in a HuggingFace repo (GeoRSCLIP, RemoteCLIP).

Both follow the same recipe: build a standard open_clip CLIP of a given
architecture, then load a domain-specific state dict downloaded with
``huggingface_hub.hf_hub_download``. They expose the project's
``ImageTextEncoder`` protocol.
"""
from __future__ import annotations

from typing import List, Optional, Union

from src import _cache  # noqa: F401  sets HF_HOME before huggingface_hub
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _load_state_dict_flexible(model, ckpt_path: str) -> None:
    """Tolerant load: handles ``{'state_dict': ...}`` wrappers, ``module.``
    prefixes, and partial matches (strict=False)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # CLIP logit_scale / position-id buffers are commonly absent — only worry
    # if entire towers failed to load.
    critical = [m for m in missing if m.startswith(("visual.", "transformer."))]
    if critical:
        raise RuntimeError(
            f"Checkpoint missing critical weights (e.g. {critical[:3]} ...). "
            "Architecture/checkpoint mismatch."
        )


class OpenClipHFEncoder:
    """Concrete encoders set ``name``, ``embed_dim``, ``_arch``,
    ``_hf_repo``, ``_hf_file``."""

    name: str = "openclip"
    embed_dim: int = 512
    image_input_size: int = 224
    _arch: str = "ViT-B-32"
    _hf_repo: str = ""
    _hf_file: str = ""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"open-clip-torch is required for {self.name}. "
                "Install: pip install open-clip-torch"
            ) from exc
        from huggingface_hub import hf_hub_download

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Loading {self.name}: {self._hf_repo}/{self._hf_file} "
              f"(arch {self._arch})")
        model, _, preprocess = open_clip.create_model_and_transforms(self._arch)
        ckpt_path = hf_hub_download(
            repo_id=self._hf_repo, filename=self._hf_file, cache_dir=cache_dir
        )
        _load_state_dict_flexible(model, ckpt_path)

        self._model = model.to(self.device).eval()
        for p in self._model.parameters():
            p.requires_grad = False
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self._arch)

    # ------------------------------------------------------------------
    def encode_text(
        self, texts: Union[str, List[str]], batch_size: int = 32
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                tok = self._tokenizer(texts[i:i + batch_size]).to(self.device)
                f = F.normalize(self._model.encode_text(tok), dim=-1)
                out.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    def encode_image(
        self, images: Union[Image.Image, List[Image.Image]], batch_size: int = 32
    ) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]
        out: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                px = torch.stack(
                    [self._preprocess(im) for im in images[i:i + batch_size]]
                ).to(self.device)
                f = F.normalize(self._model.encode_image(px), dim=-1)
                out.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    # ------------------------------------------------------------------
    def _patch_tokens(self, px: torch.Tensor) -> Optional[torch.Tensor]:
        """Replicate open_clip VisionTransformer forward, returning per-patch
        features projected into the shared space ``[1, N, D]``. Returns None
        if the visual tower is not a standard open_clip ViT."""
        v = self._model.visual
        try:
            x = v.conv1(px)                                  # [1, w, gh, gw]
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls = v.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], dim=1)
            x = x + v.positional_embedding.to(x.dtype)
            x = v.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = v.transformer(x)
            x = x.permute(1, 0, 2)
            x = v.ln_post(x)
            if getattr(v, "proj", None) is not None:
                x = x @ v.proj
            return x[:, 1:, :]                                # drop CLS
        except Exception:
            return None

    def encode_image_patches(self, image: Image.Image) -> Optional[np.ndarray]:
        """Per-patch embeddings projected into the shared space, L2-normalised,
        as a ``[n_patches, D]`` float32 array (raw cosine-comparable — no per-image
        min-max, unlike ``compute_patch_text_similarity``). ``None`` if the visual
        tower is not a standard open_clip ViT. Used by patch-level retrieval."""
        with torch.no_grad():
            px = self._preprocess(image).unsqueeze(0).to(self.device)
            patches = self._patch_tokens(px)
            if patches is None:
                return None
            pf = F.normalize(patches, dim=-1).squeeze(0)   # [N, D]
            return pf.cpu().numpy().astype(np.float32)

    def compute_patch_text_similarity(
        self, image: Image.Image, text: str
    ) -> np.ndarray:
        with torch.no_grad():
            px = self._preprocess(image).unsqueeze(0).to(self.device)
            patches = self._patch_tokens(px)
            tok = self._tokenizer([text]).to(self.device)
            tfeat = F.normalize(self._model.encode_text(tok), dim=-1)  # [1, D]
            if patches is None:
                f = F.normalize(self._model.encode_image(px), dim=-1)
                return np.array([[float((f @ tfeat.t()).item())]], np.float32)
            pf = F.normalize(patches, dim=-1)                          # [1,N,D]
            sims = (pf @ tfeat.t()).squeeze(0).squeeze(-1)             # [N]
            n = sims.shape[0]
            side = int(round(n ** 0.5))
            if side * side != n:
                return np.array([[float(sims.mean().item())]], np.float32)
            grid = sims.view(side, side).cpu().numpy().astype(np.float32)
        lo, hi = float(grid.min()), float(grid.max())
        if hi - lo < 1e-8:
            return np.zeros_like(grid, dtype=np.float32)
        return ((grid - lo) / (hi - lo)).astype(np.float32)
