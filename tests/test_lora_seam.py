"""
Fast contract tests for the encoder-agnostic LoRA seam (no peft / CLIP / GPU).

Guards the H2 fix: the LoRA trainer must drive everything off
``encoder.lora_visual_spec()`` and fail with a clear error on encoders that do
not advertise it (previously it reached into open_clip-only ``_model.visual`` and
crashed with an obscure AttributeError on clip_vitl14).
"""
from __future__ import annotations

import pytest
import torch.nn as nn

from src.encoders.base import LoRAVisualSpec
from src.lora_train import LoRAConfig, _require_lora_spec


def test_lora_config_target_modules_defaults_to_none():
    # None => the encoder's own spec supplies target modules (c_fc/c_proj vs fc1/fc2).
    assert LoRAConfig().target_modules is None


def test_require_lora_spec_raises_for_unsupported_encoder():
    class _NoLoRA:
        name = "nolora"

    with pytest.raises(TypeError, match="does not support visual LoRA"):
        _require_lora_spec(_NoLoRA())


def test_require_lora_spec_returns_spec_for_supported_encoder():
    module = nn.Linear(4, 4)

    class _HasLoRA:
        name = "haslora"

        def lora_visual_spec(self):
            return LoRAVisualSpec(
                module=module,
                target_modules=["fc1", "fc2"],
                preprocess=lambda im: im,
                forward=lambda mod, px: px,
                set_module=lambda m: None,
                to_device=lambda dev: None,
            )

    spec = _require_lora_spec(_HasLoRA())
    assert spec.module is module
    assert spec.target_modules == ["fc1", "fc2"]
