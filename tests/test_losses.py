"""Regression tests for the corrected contrastive losses.

Guards two fixed bugs:
- ``lora_train._infonce_loss`` previously used a single diagonal target while
  leaving same-caption positives in the denominator (they fought each other).
  It must now equal ``train._masked_infonce`` (mean log-prob over the positive set).
- ``model.InfoNCELoss`` previously targeted the NEGATIVE block when negatives
  were supplied. The matched target must be the positive block.
"""
import torch

from src.lora_train import _infonce_loss
from src.model import InfoNCELoss
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


def test_infonce_targets_positive_block_not_negative():
    torch.manual_seed(0)
    B, D = 4, 16
    pos = torch.randn(B, D)
    neg = torch.randn(B, D)
    crit = InfoNCELoss(temperature=0.1)
    # anchors aligned with the POSITIVES -> low loss (positives are the target)
    loss_good, _ = crit(torch.cat([pos, pos], 0), pos, negative=neg)
    # anchors aligned with the NEGATIVES -> high loss
    loss_bad, _ = crit(torch.cat([neg, neg], 0), pos, negative=neg)
    assert float(loss_good) < float(loss_bad)
