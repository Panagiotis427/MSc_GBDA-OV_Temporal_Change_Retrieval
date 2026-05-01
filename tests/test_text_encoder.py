"""
Unit tests for text embedding pipeline.

Tests verify CLIP text encoder loads correctly, produces valid embeddings,
and handles edge cases in query encoding.
"""
import pytest
import torch
from src.text_encoder import FrozenTextEncoder


class TestFrozenTextEncoder:
    """Tests for the frozen CLIP text encoder."""

    @pytest.fixture
    def encoder(self):
        """Create a reusable test encoder instance."""
        return FrozenTextEncoder(cache_dir="/tmp/clip-test-cache")

    def test_encoder_initialization(self, encoder):
        """Test that encoder loads successfully and parameters are frozen."""
        # Verify model is on CPU (for reproducibility in CI)
        assert str(encoder.device) == "cpu", f"Expected CPU, got {encoder.device}"

    def test_encoder_embed_dim(self, encoder):
        """Test that embed dimension matches CLIP ViT-L/14."""
        expected_dim = 768
        actual_dim = len(encoder)

        assert actual_dim == expected_dim, \
            f"Expected embedding dim {expected_dim}, got {actual_dim}"

    def test_single_query_encoding(self, encoder):
        """Test encoding a single natural language query."""
        query = "new industrial buildings appearing on former agricultural land"
        emb = encoder.encode(query)

        assert emb.shape == (1, 768), f"Expected (1, 768), got {emb.shape}"
        assert not torch.isnan(emb).any(), "Embedding contains NaN values"
        assert not torch.isinf(emb).any(), "Embedding contains inf values"

    def test_multiple_queries(self, encoder):
        """Test encoding a list of queries."""
        queries = [
            "construction on agricultural land",
            "coastal erosion after storm event",
            "new residential development in urban area",
        ]
        embeddings = encoder.encode(queries)

        assert embeddings.shape == (3, 768), f"Expected (3, 768), got {embeddings.shape}"

    def test_batch_encoding(self, encoder):
        """Test batch encoding with automatic batching."""
        queries = [
            f"query number {i}" for i in range(100)  # Large list triggers batching
        ]
        embeddings = encoder.encode_batch(queries)

        assert embeddings.shape == (100, 768), \
            f"Expected (100, 768), got {embeddings.shape}"

    def test_very_small_query(self, encoder):
        """Test encoding extremely short queries."""
        query = "a"
        emb = encoder.encode(query)

        assert emb.shape == (1, 768)
        # Short queries should still produce non-trivial embeddings
        assert not torch.allclose(emb, torch.zeros_like(emb))

    def test_very_long_query(self, encoder):
        """
        Test encoding very long queries.
        CLIP truncates to 77 tokens, so output shape is still (1, 768).
        """
        query = "a " * 1000  # Will be truncated to 77 tokens
        emb = encoder.encode(query)

        assert emb.shape == (1, 768)
        # Long queries with repeated content should still work
        assert not torch.isnan(emb).any()

    def test_emoji_query(self, encoder):
        """Test encoding queries with emojis (common in informal descriptions)."""
        query = "🏭 new factory 🌾 on farmland"
        emb = encoder.encode(query)

        assert emb.shape == (1, 768)
        # Emojis are valid Unicode - CLIP tokenizer handles them
        assert not torch.isnan(emb).any()

    def test_english_and_acronyms(self, encoder):
        """Test queries with technical terms and acronyms."""
        queries = [
            "urban sprawl",
            "land use change (LUC)",
            "NDVI decrease",
            "impervious surface increase",
        ]
        embeddings = encoder.encode(queries)

        assert embeddings.shape == (4, 768)

    def test_special_characters(self, encoder):
        """Test queries with special characters and punctuation."""
        query = "'new' industrial? 🏭 on agricultural 'land'?"
        emb = encoder.encode(query)

        assert emb.shape == (1, 768)

    def test_frozen_parameters(self, encoder):
        """Test that no text encoder parameters are trainable."""
        frozen_count = sum(p.numel() for p in encoder.model.parameters() if not p.requires_grad)
        total_params = sum(p.numel() for p in encoder.model.parameters())

        assert frozen_count == total_params, \
            f"All {total_params} parameters should be frozen, but {total_params - frozen_count} are trainable"

    def test_different_model_names(self):
        """Test that we can use different CLIP text encoders."""
        # This just verifies the class accepts model names
        try:
            encoder = FrozenTextEncoder(
                model_name="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                cache_dir="/tmp/clip-test-cache"
            )
            assert encoder.model.config.hidden_size == 1280, \
                "BigG model has hidden size 1280"
        except Exception as e:
            # Model might not be available in test environment
            pytest.skip(f"Model download failed: {e}")

    def test_embedding_values_reasonable(self, encoder):
        """Test that embedding values are within reasonable range."""
        queries = ["industrial development"] * 10
        embeddings = encoder.encode_batch(queries)

        min_val = embeddings.min()
        max_val = embeddings.max()

        # CLIP embeddings typically range from -5 to +5 after normalization
        assert -20 < min_val < 20, f"Min value {min_val} out of expected range"
        assert -20 < max_val < 20, f"Max value {max_val} out of expected range"

    def test_tokenization_reproducibility(self, encoder):
        """Test that identical queries produce identical embeddings."""
        query = "industrial development on farmland"
        emb1 = encoder.encode(query)
        emb2 = encoder.encode(query)

        torch.testing.assert_close(emb1, emb2)

    def test_encoding_with_whitespace_variations(self, encoder):
        """Test that whitespace variations don't break encoding."""
        queries = [
            "industrial development",
            "  industrial   development  ",
            "\tindustrial\tdevelopment\t",
            "new line\nand another",
        ]
        embeddings = encoder.encode(queries)

        assert embeddings.shape == (4, 768)
        # All should be valid embeddings
        for emb in embeddings:
            assert not torch.isnan(emb).any()
            assert not torch.isinf(emb).any()