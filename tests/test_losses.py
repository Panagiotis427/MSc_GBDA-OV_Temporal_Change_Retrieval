"""Regression tests for the corrected contrastive losses.

Guards a fixed bug:
- ``lora_train._infonce_loss`` previously used a single diagonal target while
  leaving same-caption positives in the denominator (they fought each other).
  It must now equal ``train._masked_infonce`` (mean log-prob over the positive set).
"""
import torch

from src.lora_train import _infonce_loss
from src.train import _masked_infonce


def test_lora_loss_matches_masked_infonce():
    torch.manual_seed(0)
    delta = torch.randn(6, 32)
    text = torch.randn(6, 32)
    cid = torch.tensor([0, 0, 1, 1, 2, 2])
    pos_mask = cid[:, None] == cid[None, :]
    a = _infonce_loss(delta, text, pos_mask)
    b = _masked_infonce(delta, text, pos_mask)
    assert torch.allclose(a, b, atol=1e-6)


def test_lora_loss_rewards_correct_caption_alignment():
    # Rows 0 and 1 share caption 0. When their change vectors point at caption 0's
    # text the loss must be LOWER than when they point at the wrong caption — i.e.
    # same-caption rows are treated as mutual positives, not as distractors.
    text = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cid = torch.tensor([0, 0, 1])
    pos_mask = cid[:, None] == cid[None, :]
    good = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])   # aligned to caption 0
    bad = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])    # aligned to wrong caption
    assert float(_infonce_loss(good, text, pos_mask)) < float(_infonce_loss(bad, text, pos_mask))
