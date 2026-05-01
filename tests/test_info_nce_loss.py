"""
Unit tests for InfoNCE Loss function.

Tests verify contrastive learning formulation is correct, temperature parameter works,
and gradients flow properly through the loss computation.
"""
import pytest
import torch
from src.model import InfoNCELoss


class TestInfoNCELoss:
    """Tests for the InfoNCE contrastive loss function."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create consistent test embeddings matching CLIP dimension."""
        batch_size = 8
        embed_dim = 768
        return torch.randn(2 * batch_size, embed_dim), torch.randn(batch_size, embed_dim)

    def test_basic_forward_pass(self, sample_embeddings):
        """Test that loss computation runs without errors."""
        anchor, positive = sample_embeddings
        criterion = InfoNCELoss()
        loss, logits = criterion(anchor, positive)

        assert not torch.isnan(loss).item(), f"Loss is NaN: {loss.item()}"
        assert not torch.isinf(loss).item(), f"Loss is inf: {loss.item()}"

    def test_loss_is_positive(self, sample_embeddings):
        """
        Test that InfoNCE loss is always non-negative.
        This validates the log-sum-exp formulation is correct.
        """
        anchor, positive = sample_embeddings
        criterion = InfoNCELoss()
        loss, _ = criterion(anchor, positive)

        assert loss >= 0, f"Loss should be non-negative but got {loss.item()}"

    def test_temperature_parameter(self, sample_embeddings):
        """
        Test that temperature parameter controls sharpness of distribution.
        Higher τ → flatter softmax → lower gradients.
        """
        anchor, positive = sample_embeddings

        # Standard temperature (sharp)
        criterion_standard = InfoNCELoss(temperature=0.1)
        loss_standard = criterion_standard(anchor, positive)[0]

        # Higher temperature (smoother)
        criterion_smooth = InfoNCELoss(temperature=2.0)
        loss_smooth = criterion_smooth(anchor, positive)[0]

        # With τ > 1, softmax becomes flatter → higher expected log probability
        assert abs(loss_standard - loss_smooth) > 1e-5, \
            "Temperature should affect the loss value"

    def test_different_temperatures_on_same_data(self, sample_embeddings):
        """Test that we can override temperature on-the-fly."""
        anchor = torch.randn(16, 768)
        positive = torch.randn(8, 768)

        criterion = InfoNCELoss(temperature=0.5)

        loss_0_5 = criterion(anchor, positive, temperature=0.5)[0]
        loss_2_0 = criterion(anchor, positive, temperature=2.0)[0]

        assert not torch.allclose(loss_0_5, loss_2_0), \
            "Different temperatures should produce different losses"

    def test_batch_size_auto_detection(self, sample_embeddings):
        """Test that batch size is automatically inferred."""
        anchor = torch.randn(16, 768)  # 2 views per sample → 8 samples
        positive = torch.randn(8, 768)

        criterion = InfoNCELoss()
        loss, _ = criterion(anchor, positive)

        assert not torch.isnan(loss).item(), "Batch size auto-detection should work"

    def test_empty_input_handling(self):
        """Test that empty inputs raise appropriate errors."""
        try:
            anchor = torch.randn(0, 768)  # Empty batch
            positive = torch.randn(0, 768)
            criterion = InfoNCELoss()
            loss, _ = criterion(anchor, positive)
            assert False, "Should have raised an error for empty input"
        except Exception as e:
            assert "dimension" in str(e).lower() or "empty" in str(e).lower()

    def test_gradient_flow(self):
        """
        Test that gradients flow correctly through the loss.
        Critical for backpropagation during training.
        """
        anchor = torch.randn(8, 768, requires_grad=True)
        positive = torch.randn(4, 768, requires_grad=True)

        criterion = InfoNCELoss()
        loss, logits = criterion(anchor, positive)
        loss.backward()

        assert not torch.isnan(anchor.grad).any(), "Anchor gradients should be finite"
        assert not torch.isnan(positive.grad).any(), "Positive gradients should be finite"

    def test_similarities_computation(self):
        """Test that cosine similarity is computed correctly."""
        criterion = InfoNCELoss()

        # Create two identical embeddings - cosine sim should be 1
        x = torch.randn(4, 768)
        sim_matrix = criterion._compute_similarities(x, x)

        assert (sim_matrix.max().item() > 0.9), \
            "Identical vectors should have high cosine similarity"
        # Check that min value is negative (random embeddings aren't all positively correlated)
        assert sim_matrix.min().item() < 0, \
            "Random embeddings should have some negative similarities"

    def test_temperature_extremes(self):
        """
        Test behavior at temperature extremes.
        Very low τ → almost one-hot softmax (hard)
        Very high τ → uniform distribution (easy, near-zero gradients)
        """
        anchor = torch.randn(8, 768)
        positive = torch.randn(4, 768)

        # Extreme temperature: loss should still be finite
        criterion_extreme = InfoNCELoss(temperature=10.0)
        loss_extreme = criterion_extreme(anchor, positive)[0]
        assert not torch.isnan(loss_extreme).item()
        assert abs(loss_extreme) < 5.0  # Very high τ should give small loss
