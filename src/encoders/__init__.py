"""
Encoder subpackage.

Public API:

    from src.encoders import get_encoder, ImageTextEncoder
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from .base import ImageTextEncoder


_FACTORIES: Dict[str, Callable[..., ImageTextEncoder]] = {}


def register_encoder(name: str, factory: Callable[..., ImageTextEncoder]) -> None:
    _FACTORIES[name] = factory


def get_encoder(name: str, **kwargs: Any) -> ImageTextEncoder:
    if name not in _FACTORIES:
        raise ValueError(
            f"Unknown encoder: '{name}'. Registered: {sorted(_FACTORIES)}"
        )
    return _FACTORIES[name](**kwargs)


def list_encoders() -> list[str]:
    return sorted(_FACTORIES)


def _clip_vitl14_factory(**kwargs: Any) -> ImageTextEncoder:
    from .clip_vitl14 import CLIPViTL14Encoder
    return CLIPViTL14Encoder(**kwargs)


def _georsclip_factory(**kwargs: Any) -> ImageTextEncoder:
    from .georsclip import GeoRSCLIPEncoder
    return GeoRSCLIPEncoder(**kwargs)


def _remoteclip_factory(**kwargs: Any) -> ImageTextEncoder:
    from .remoteclip import RemoteCLIPEncoder
    return RemoteCLIPEncoder(**kwargs)


register_encoder("clip_vitl14", _clip_vitl14_factory)
register_encoder("georsclip", _georsclip_factory)
register_encoder("remoteclip", _remoteclip_factory)


__all__ = ["ImageTextEncoder", "get_encoder", "register_encoder", "list_encoders"]
