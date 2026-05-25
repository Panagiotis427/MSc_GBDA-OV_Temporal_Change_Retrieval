"""
Unit tests for change feature engineering module.

Tests verify tensor shapes, computation correctness, and edge case handling
for the delta_f calculator in src/features.py.
"""
import pytest
import torch
from src.features import (
    compute_change_feature,
    compute_change_magnitude,
    normalize_embeddings
)


class TestComputeChangeFeature:
    """Tests for the main change feature computation."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create consistent test data."""
        batch_size = 4
        embed_dim = 128
        return (
            torch.randn(batch_size, embed_dim),
            torch.randn(batch_size, embed_dim)
        )

    def test_difference_mode_shape(self, sample_embeddings):
        """Test that difference mode outputs correct shape."""
        emb_t1, emb_t2 = sample_embeddings
        delta = compute_change_feature(emb_t1, emb_t2, mode="difference")

        assert delta.shape == (4, 128), f"Expected (4, 128), got {delta.shape}"

    def test_difference_mode_values(self, sample_embeddings):
        """Test that delta_f = f_T2 - f_T1 is numerically correct."""
        emb_t1, emb_t2 = sample_embeddings
        expected = emb_t2 - emb_t1
        actual = compute_change_feature(emb_t1, emb_t2, mode="difference")

        torch.testing.assert_close(actual, expected)

    def test_concatenate_mode_shape(self, sample_embeddings):
        """Test that concatenation mode doubles the embedding dimension."""
        emb_t1, emb_t2 = sample_embeddings
        delta = compute_change_feature(emb_t1, emb_t2, mode="concatenate")

        assert delta.shape == (4, 256), f"Expected (4, 256), got {delta.shape}"

    def test_concatenate_mode_values(self, sample_embeddings):
        """Test that concatenation interleaves tensors correctly."""
        emb_t1, emb_t2 = sample_embeddings
        expected = torch.cat([emb_t1, emb_t2], dim=-1)
        actual = compute_change_feature(emb_t1, emb_t2, mode="concatenate")

        torch.testing.assert_close(actual, expected)

    def test_invalid_mode_raises(self):
        """Test that unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            compute_change_feature(
                torch.randn(4, 128),
                torch.randn(4, 128),
                mode="invalid"
            )

    def test_mismatched_shapes_raises(self):
        """Test that mismatched embedding shapes raise AssertionError."""
        emb_t1 = torch.randn(4, 128)
        emb_t2 = torch.randn(4, 256)  # Different dimension

        with pytest.raises(AssertionError):
            compute_change_feature(emb_t1, emb_t2)

    def test_single_sample(self):
        """Test edge case: single sample in batch."""
        emb_t1 = torch.randn(1, 768)  # CLIP ViT-L/14 dimension
        emb_t2 = torch.randn(1, 768)
        delta = compute_change_feature(emb_t1, emb_t2, mode="difference")

        assert delta.shape == (1, 768)

    def test_large_batch(self):
        """Test edge case: large batch size."""
        batch_size = 1024
        embed_dim = 1024
        emb_t1 = torch.randn(batch_size, embed_dim)
        emb_t2 = torch.randn(batch_size, embed_dim)
        delta = compute_change_feature(emb_t1, emb_t2, mode="difference")

        assert delta.shape == (batch_size, embed_dim)

    def test_zero_difference(self):
        """Test that identical embeddings produce zero delta."""
        emb = torch.randn(4, 128)
        delta = compute_change_feature(emb, emb, mode="difference")

        assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)

    def test_very_small_difference(self):
        """Test numerical stability with tiny changes."""
        emb_t1 = torch.randn(4, 768)
        emb_t2 = emb_t1 + 1e-9 * torch.randn_like(emb_t1)
        delta = compute_change_feature(emb_t1, emb_t2, mode="difference")

        assert delta.shape == (4, 768)
        # Delta should equal the actual difference between embeddings
        expected = emb_t2 - emb_t1
        torch.testing.assert_close(delta, expected)

    def test_very_large_difference(self):
        """Test numerical stability with large changes."""
        emb_t1 = torch.randn(4, 768) * 10
        emb_t2 = torch.randn(4, 768) * 10 + 1000  # Large offset
        delta = compute_change_feature(emb_t1, emb_t2, mode="difference")

        assert torch.allclose(delta, emb_t2 - emb_t1, atol=1e-5)

    def test_dtype_preservation(self):
        """Test that dtype is preserved through computation."""
        # Test with float32 (default)
        emb_t1 = torch.randn(4, 768, dtype=torch.float32)
        delta_f32 = compute_change_feature(emb_t1, emb_t1 + 1e-3)
        assert delta_f32.dtype == torch.float32

        # Test with float64 (for numerical precision tests)
        emb_t1 = torch.randn(4, 768, dtype=torch.float64)
        delta_f64 = compute_change_feature(emb_t1, emb_t1 + 1e-9)
        assert delta_f64.dtype == torch.float64


class TestComputeChangeMagnitude:
    """Tests for L2 norm computation."""

    def test_output_shape(self):
        """Test that magnitude returns batch-dimension tensor."""
        emb_t1 = torch.randn(8, 768)
        emb_t2 = torch.randn(8, 768) + 0.5
        magnitudes = compute_change_magnitude(emb_t1, emb_t2)

        assert magnitudes.shape == (8,), f"Expected (8,), got {magnitudes.shape}"

    def test_magnitude_positive(self):
        """Test that magnitude is always non-negative."""
        for _ in range(10):
            emb_t1 = torch.randn(4, 768)
            emb_t2 = torch.randn(4, 768) + torch.randn_like(emb_t1) * 0.1
            mag = compute_change_magnitude(emb_t1, emb_t2)
            assert (mag >= 0).all(), "Magnitude should be non-negative"

    def test_zero_change(self):
        """Test that zero change produces magnitude of zero."""
        emb = torch.randn(4, 768)
        mag = compute_change_magnitude(emb, emb)
        assert torch.allclose(mag, torch.zeros_like(mag), atol=1e-6)

    def test_large_change(self):
        """Test magnitude with large changes."""
        emb_t1 = torch.randn(4, 768) * 10
        emb_t2 = torch.randn(4, 768) * 10 + 100
        mag = compute_change_magnitude(emb_t1, emb_t2)
        assert (mag > 50).all(), "Large change should produce large magnitude"


class TestNormalizeEmbeddings:
    """Tests for L2 normalization."""

    @pytest.fixture
    def sample_embeddings(self):
        return torch.randn(4, 768)

    def test_output_shape_preserved(self, sample_embeddings):
        """Test that shape is preserved after normalization."""
        emb = sample_embeddings
        normalized = normalize_embeddings(emb)

        assert normalized.shape == emb.shape

    def test_unit_norm(self, sample_embeddings):
        """Test that each row has L2 norm of 1."""
        emb = sample_embeddings
        normalized = normalize_embeddings(emb)

        norms = torch.norm(normalized, p=2, dim=-1)
        # Use allclose for floating point comparison (tolerates tiny numerical errors)
        assert torch.allclose(norms, torch.ones_like(norms)), f"Expected all ones, got {norms.unique()}"

    def test_division_by_zero_handled(self):
        """Test that zero-norm embeddings don't cause NaN/inf."""
        emb = torch.zeros(4, 768)  # All zeros -> norm = 0
        normalized = normalize_embeddings(emb)

        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()

    def test_cosine_similarity_unchanged(self, sample_embeddings):
        """
        Test that normalizing doesn't change cosine similarity.
        This validates the normalization is mathematically correct.
        """
        emb_a = torch.randn(10, 768)
        emb_b = torch.randn(10, 768)

        # Both computations use the numerically stable matmul of normalized vectors
        # This avoids floating point errors from element-wise division
        sim_before = torch.matmul(
            torch.nn.functional.normalize(emb_a, dim=-1),
            torch.nn.functional.normalize(emb_b, dim=-1).T
        )

        # Normalized embeddings using our function
        norm_a = normalize_embeddings(emb_a)
        norm_b = normalize_embeddings(emb_b)

        # Cosine similarity after normalization (should be identical to stable computation)
        sim_after = norm_a @ norm_b.T

        torch.testing.assert_close(sim_before, sim_after, rtol=1e-5, atol=1e-6)
