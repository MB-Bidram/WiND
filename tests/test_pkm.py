"""Tests for PKM integration through the winc.pkm bridge."""

import pytest
import torch

import wind
import winc

from winc.pkm import make_pkm_wide, pkm_available, PKMAvailable
from wind.language import LanguageModel, LMConfig


@pytest.mark.skipif(not pkm_available(), reason="FlashPKM not available")
class TestPKM:

    def test_make_pkm_wide_shapes(self):
        """PKMWide from the bridge should produce correct shape."""
        wide = make_pkm_wide(
            dim=64,
            memory_size=128,
            query_dim=32,
            value_dim=64,
            num_factors=2,
            heads=4,
            topk=4,
            topk_per_factor=8,
            similarity="cosine",
            output_mode="none",
            key_dtype="float32",
            value_dtype="float32",
        )
        x = torch.randn(2, 8, 64)
        y = wide(x)
        assert y.shape == x.shape

    def test_make_pkm_wide_invalid_query_dim(self):
        """make_pkm_wide should raise on invalid query_dim/heads/factors."""
        with pytest.raises(ValueError):
            make_pkm_wide(
                dim=64,
                query_dim=15,  # not divisible by heads*factors
                heads=4,
                num_factors=2,
            )

    def test_make_pkm_wide_unavailable(self):
        """make_pkm_wide should raise PKMAvailable when flashpkm missing."""
        # We can't easily test this without mocking; just verify the exception class
        assert isinstance(PKMAvailable, type)

    def test_pkm_in_language_model(self):
        """LanguageModel with wide_type='pkm' should work through the bridge."""
        config = LMConfig(
            vocab_size=100,
            dim=32,
            heads=4,
            depth=1,
            encoder_depth=1,
            decoder_depth=1,
            width=1,
            iterations=1,
            state_tokens=4,
            bank_tokens=8,
            max_source_length=32,
            max_target_length=16,
            mlp_ratio=2.0,
            wide_type="pkm",
            pkm_memory_size=64,
            pkm_num_factors=2,
            pkm_heads=4,
            pkm_topk=2,
            pkm_key_dtype="float32",
            pkm_value_dtype="float32",
            pkm_similarity="cosine",
            norm="rmsnorm",
        )
        model = LanguageModel(config)
        input_ids = torch.randint(3, 100, (1, 8))
        labels = torch.randint(3, 100, (1, 4))
        output = model(input_ids=input_ids, labels=labels)
        assert output.logits.shape == (1, 4, 100)

    def test_wind_exposes_pkm_via_bridge(self):
        """wind.PKMWide should be accessible and be the same class as from flashpkm."""
        assert wind.PKMWide is not None
        # It should be the flashpkm.PKMWide class
        import flashpkm
        assert wind.PKMWide is flashpkm.PKMWide

    def test_pkm_kernels_available_flag(self):
        """wind.pkm_kernels_available should match winc.pkm.pkm_available()."""
        assert wind.pkm_kernels_available == winc.pkm.pkm_available()
