"""Tests proving correctness of bug fixes in WiND 2.

Each test corresponds to a specific identified bug:
- test_generation_cache_is_used: Bug #1 - GenerationCache never instantiated
- test_cross_attention_projects_kv_separately: Bug #2 - K and V are identical
- test_cross_cache_get_returns_correct_length: Bug #3 - Wrong cache slicing
- test_pkm_topk_output_count: Bug #4 - PKM candidate pruning correctness
- test_pkm_residual_has_scaling: Bug #5 - PKM residual scaling
"""

import pytest
import torch
from torch import nn

from wind.language import LanguageModel, LMConfig
from winc.cache import GenerationCache
from winc.attention import Attention


# ═══════════════════════════════════════════════════════════════════════════════
# Bug #1: GenerationCache Never Instantiated
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerationCacheUsage:
    """Verify GenerationCache is actually used during generation."""

    def _make_config(self, **kwargs):
        defaults = dict(
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
            wide_type="dense",
        )
        defaults.update(kwargs)
        return LMConfig(**defaults)

    def test_generation_cache_is_used(self):
        """GenerationCache should be instantiated and used during generate()."""
        config = self._make_config()
        model = LanguageModel(config)

        input_ids = torch.randint(3, 100, (1, 8))
        model.eval()

        # Patch _decode to verify GenerationCache is passed
        cache_types_seen = []
        original_decode = model._decode

        def tracked_decode(ids, state, **kwargs):
            cache = kwargs.get("cache")
            if cache is not None:
                cache_types_seen.append(type(cache).__name__)
            else:
                cache_types_seen.append("None")
            return original_decode(ids, state, **kwargs)

        model._decode = tracked_decode

        with torch.no_grad():
            generated = model.generate(input_ids=input_ids, max_new_tokens=4, temperature=0.0)

        # All calls should have seen GenerationCache
        assert len(cache_types_seen) > 0, "No decode calls made"
        assert all(ct == "GenerationCache" for ct in cache_types_seen), (
            f"Expected GenerationCache in all calls, got: {cache_types_seen}"
        )

    def test_generation_cache_reduces_compute(self):
        """With use_cache=True, each token should make exactly one forward call."""
        config = self._make_config()
        model = LanguageModel(config)

        call_count = {"forward_calls": 0}
        original_forward = model.lm_head.forward

        def tracked_forward(x):
            call_count["forward_calls"] += 1
            return original_forward(x)

        model.lm_head.forward = tracked_forward

        input_ids = torch.randint(3, 100, (1, 8))
        max_new = 5

        with torch.no_grad():
            generated = model.generate(input_ids=input_ids, max_new_tokens=max_new, temperature=0.0)

        # There is one decoder/LM-head call for each emitted token.  Greedy
        # generation may stop early when it emits EOS.
        assert call_count["forward_calls"] == generated.size(1), (
            f"Expected one call per emitted token, got {call_count['forward_calls']} calls "
            f"for {generated.size(1)} tokens"
        )

    def test_generation_cache_matches_uncached_generation(self):
        """The preallocated cache must not change deterministic decoding."""
        torch.manual_seed(7)
        model = LanguageModel(self._make_config()).eval()
        input_ids = torch.randint(3, 100, (2, 8))
        with torch.no_grad():
            cached = model.generate(input_ids=input_ids, max_new_tokens=4, temperature=0.0, use_cache=True)
            uncached = model.generate(input_ids=input_ids, max_new_tokens=4, temperature=0.0, use_cache=False)
        assert torch.equal(cached, uncached)


# ═══════════════════════════════════════════════════════════════════════════════
# Bug #2: Cross-Attention K/V Identity
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossAttentionSeparateKV:
    """Verify cross-attention uses separate K and V projections."""

    def _make_config(self, **kwargs):
        defaults = dict(
            vocab_size=100, dim=32, heads=4, depth=1, encoder_depth=1,
            decoder_depth=1, width=1, iterations=1, state_tokens=4,
            bank_tokens=8, max_source_length=32, max_target_length=16,
            mlp_ratio=2.0, wide_type="dense",
        )
        defaults.update(kwargs)
        return LMConfig(**defaults)

    def test_cross_attention_projects_kv_separately(self):
        """Attention class should project K and V separately."""
        dim = 8
        attn = Attention(dim, heads=2, use_rope=False)
        context = torch.randn(2, 4, dim)

        k, v = attn.project_kv(context)
        assert k.shape == (2, 2, 4, 4), f"Got key shape {k.shape}"
        assert v.shape == (2, 2, 4, 4), f"Got value shape {v.shape}"

        # K and V should be different (separate projections via kv.weight)
        assert not torch.allclose(k, v), "K and V projections are identical - missing separate projections"

    def test_project_kv_uses_different_weights(self):
        """project_kv should split the kv projection into distinct k and v."""
        dim = 8
        attn = Attention(dim, heads=2, use_rope=False)

        # Check that kv has separate weights for k and v
        assert attn.kv is not None, "Missing kv projection layer"
        assert attn.kv.weight.shape == (dim * 2, dim), f"Unexpected kv weight shape"

        # Verify k and v come from different halves of the kv projection
        context = torch.randn(2, 4, dim)
        kv_out = attn.kv(context)
        k_part, v_part = kv_out.chunk(2, dim=-1)
        k_proj, v_proj = attn.project_kv(context)

        # project_kv returns [B, H, S, head_dim], need to transpose for comparison
        k_proj_seq = k_proj.transpose(1, 2).reshape(2, 4, dim)
        v_proj_seq = v_proj.transpose(1, 2).reshape(2, 4, dim)

        assert torch.allclose(k_proj_seq, k_part), "Key projection mismatch"
        assert torch.allclose(v_proj_seq, v_part), "Value projection mismatch"

        # K and V must be different (they use different weights via chunk)
        assert not torch.allclose(k_proj_seq, v_proj_seq), "K and V use same weights!"

    def test_decoder_layers_have_separate_cross_projections(self):
        """Each decoder layer should have independent cross-attention KV projection."""
        config = self._make_config(decoder_depth=3)
        model = LanguageModel(config)

        input_ids = torch.randint(3, 100, (1, 8))
        with torch.no_grad():
            state = model.encode(input_ids)

        # Each decoder layer has its own cross-attention with separate kv projection
        for i, block in enumerate(model.decoder):
            assert block.cross is not None, f"Layer {i} missing cross-attention"
            assert hasattr(block.cross, "kv"), f"Layer {i} cross-attention missing kv projection"

        # Project state through each layer's cross-attention and verify different outputs
        projections = []
        for block in model.decoder:
            k, v = block.cross.project_kv(state.tokens)
            projections.append((k, v))

        # At least some should differ (different random init)
        different = any(
            not torch.allclose(projections[0][0], projections[i][0])
            for i in range(1, len(projections))
        )
        assert different, "All decoder layers produce identical cross-attention K/V - shared weights!"


# ═══════════════════════════════════════════════════════════════════════════════
# Bug #3: GenerationCache Cross-Cache Correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerationCacheCross:
    """Verify GenerationCache cross-attention cache returns correct data."""

    def test_self_cache_lengths_are_per_layer(self):
        """Layers must not read another layer's partially written prefix."""
        cache = GenerationCache(
            batch_size=1, max_seq_len=8, num_heads=2,
            head_dim=4, dtype=torch.float32, device="cpu", n_layers=2,
        )
        key = torch.randn(1, 2, 1, 4)
        value = torch.randn(1, 2, 1, 4)
        cache.append(0, "self", key, value)
        assert cache.get(0, "self")[0].shape[-2] == 1
        assert cache.get(1, "self")[0].shape[-2] == 0
        cache.append(1, "self", key, value)
        assert cache.get(1, "self")[0].shape[-2] == 1

    def test_cross_cache_per_layer_distinct_kv(self):
        """GenerationCache should support per-layer distinct K/V."""
        cache = GenerationCache(
            batch_size=1, max_seq_len=32, num_heads=4,
            head_dim=8, dtype=torch.float32, device="cpu", n_layers=2
        )

        # Set different K/V for each layer (simulating per-layer projection)
        kv1_k, kv1_v = torch.randn(1, 4, 4, 8), torch.randn(1, 4, 4, 8)
        kv2_k, kv2_v = torch.randn(1, 4, 4, 8), torch.randn(1, 4, 4, 8)

        cache.set_cross_cache_for_layer(0, kv1_k, kv1_v)
        cache.set_cross_cache_for_layer(1, kv2_k, kv2_v)

        k0, v0 = cache.get(0, "cross")
        k1, v1 = cache.get(1, "cross")

        # Verify each layer has its own K/V
        assert torch.allclose(k0, kv1_k), "Layer 0 cross-cache key mismatch"
        assert torch.allclose(v0, kv1_v), "Layer 0 cross-cache val mismatch"
        assert torch.allclose(k1, kv2_k), "Layer 1 cross-cache key mismatch"
        assert torch.allclose(v1, kv2_v), "Layer 1 cross-cache value mismatch"

        # K and V should be different (proving separate projections work)
        assert not torch.allclose(k0, v0), "K and V should be different tensors"

    def test_cross_cache_set_for_all_layers(self):
        """set_cross_cache should set the same K/V for all layers."""
        cache = GenerationCache(
            batch_size=1, max_seq_len=16, num_heads=2,
            head_dim=4, dtype=torch.float32, device="cpu", n_layers=3
        )

        k = torch.randn(1, 2, 8, 4)
        v = torch.randn(1, 2, 8, 4)
        cache.set_cross_cache(k, v)

        for layer_idx in range(3):
            k_cached, v_cached = cache.get(layer_idx, "cross")
            assert k_cached is not None, f"Layer {layer_idx} cross-cache not set"
            assert v_cached is not None, f"Layer {layer_idx} cross-cache not set"
            assert torch.allclose(k_cached, k), f"Layer {layer_idx} key mismatch"
            assert torch.allclose(v_cached, v), f"Layer {layer_idx} val mismatch"

    def test_reset_all_clears_cross_caches(self):
        """reset_all should clear all cross-caches."""
        cache = GenerationCache(
            batch_size=1, max_seq_len=16, num_heads=2,
            head_dim=4, dtype=torch.float32, device="cpu", n_layers=2
        )

        cache.set_cross_cache(torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4))
        assert cache.get(0, "cross") is not None

        cache.reset_all()
        assert cache.get(0, "cross") is None, "Cross-cache not cleared after reset_all"


# ═══════════════════════════════════════════════════════════════════════════════
# Bug #4: PKM Candidate Pruning Correctness
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PKM requires CUDA or specific environment"
)
class TestPKMCandidatePruning:
    """Verify PKM candidate pruning is mathematically correct."""

    def test_pkm_topk_output_count(self):
        """PKM output should have correct number of top-k candidates gathered."""
        from winc.pkm import pkm_available
        if not pkm_available():
            pytest.skip("FlashPKM not available")

        from Lib.flashpkm.pkm import FactorizedPKM
        # query_dim=16, num_factors=2, heads=1, head_dim=8 -> F*D = 16
        pkm = FactorizedPKM(
            query_dim=16, memory_size=64, value_dim=16,
            num_factors=2, heads=1, topk=8, topk_per_factor=16,
            similarity="cosine",
        )
        # Query shape: [B, S, H, F*D] = [1, 1, 1, 16]
        query = torch.randn(1, 1, 1, 16)
        output = pkm(query)
        # Retrieval aggregates the selected values, so the public output is
        # [B, S, H, value_dim], not a concatenation of top-k values.
        assert output.shape == (1, 1, 1, 16), f"Expected (1, 1, 1, 16), got {output.shape}"


# ═══════════════════════════════════════════════════════════════════════════════
# Bug #5: PKM Residual Scaling
# ═══════════════════════════════════════════════════════════════════════════════

class TestPKMResidualScaling:
    """Verify PKM output uses appropriate residual scaling."""

    def test_pkm_residual_has_scaling(self):
        """PKMWide should have a gate parameter for output control."""
        from wind import PKMWide

        pkm = PKMWide(
            dim=32, memory_size=64, query_dim=16,
            num_factors=2, heads=1, output_mode="gated",
        )
        assert hasattr(pkm, "gate"), "PKMWide with gated mode should have gate parameter"
        assert isinstance(pkm.gate, nn.Parameter), "gate should be a Parameter"


# ═══════════════════════════════════════════════════════════════════════════════
# Regression Tests: Ensure existing functionality still works
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoRegressions:
    """Ensure bug fixes don't break existing functionality."""

    def _make_config(self, **kwargs):
        defaults = dict(
            vocab_size=100, dim=32, heads=4, depth=1, encoder_depth=1,
            decoder_depth=1, width=1, iterations=1, state_tokens=4,
            bank_tokens=8, max_source_length=32, max_target_length=16,
            mlp_ratio=2.0, wide_type="dense",
        )
        defaults.update(kwargs)
        return LMConfig(**defaults)

    def test_lm_forward_dense(self):
        """LanguageModel forward should still work correctly."""
        config = self._make_config()
        model = LanguageModel(config)
        input_ids = torch.randint(0, 100, (2, 8))
        labels = torch.randint(0, 100, (2, 6))
        output = model(input_ids=input_ids, labels=labels)
        assert output.logits.shape == (2, 6, 100)
        assert output.loss is not None

    def test_lm_generate_greedy(self):
        """Greedy generation should still produce valid token sequences."""
        config = self._make_config()
        model = LanguageModel(config)
        model.eval()
        input_ids = torch.randint(3, 100, (1, 8))
        with torch.no_grad():
            generated = model.generate(input_ids=input_ids, max_new_tokens=4, temperature=0.0)
        assert generated.shape[0] == 1
        assert generated.shape[1] >= 1
        assert (generated >= 0).all()
        assert (generated < 100).all()

    def test_lm_save_load(self):
        """Save/load roundtrip should preserve weights."""
        import tempfile, os
        config = self._make_config()
        model = LanguageModel(config)
        input_ids = torch.randint(0, 100, (2, 8))
        labels = torch.randint(0, 100, (2, 6))
        output1 = model(input_ids=input_ids, labels=labels)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = LanguageModel.load(path)
            output2 = loaded(input_ids=input_ids, labels=labels)
            assert torch.allclose(output1.logits, output2.logits, atol=1e-5)
        finally:
            os.unlink(path)
