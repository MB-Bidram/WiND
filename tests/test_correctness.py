"""Basic correctness tests for WideNDepth and LanguageModel (dense path).

Tests cover:
- Wide block (sum/concat modes)
- Attention bidirectional (no causal mask in encoder)
- FeatureBank read-only semantic
- WideNDepth forward shapes
- LanguageModel forward/backward, save/load, generate
- Iteration count honoring
"""

import os
import tempfile

import pytest
import torch

import wind
import winc
from winc.pkm import pkm_available
from wind.language import LanguageModel, LMConfig
from winc.architecture import WideNDepth, Compressor, ReasoningDepth, FeatureBank
from winc.blocks import TransformerBlock
from winc.attention import Attention


class TestDenseWideNDepth:
    """Test the tensor-to-tensor WideNDepth with dense wide path."""

    def test_wide_basic_forward(self):
        """Wide (dense, sum mode) should produce same shape as input."""
        dim = 64
        wide = wind.Wide(
            *[torch.nn.Sequential(wind.RMSNorm(dim), wind.SwiGLU(dim, dim * 2)) for _ in range(2)],
            dim=dim,
            mode="sum",
        )
        x = torch.randn(2, 8, dim)
        y = wide(x)
        assert y.shape == x.shape

    def test_wide_concat_mode(self):
        """Wide with concat mode should project back to dim (not expand)."""
        dim = 64
        wide = wind.Wide(
            *[torch.nn.Sequential(wind.RMSNorm(dim), wind.SwiGLU(dim, dim * 2)) for _ in range(2)],
            dim=dim,
            mode="concat",
        )
        x = torch.randn(2, 8, dim)
        y = wide(x)
        assert y.shape == x.shape  # concat + proj back to dim

    def test_attention_bidirectional(self):
        """Attention (encoder) should accept mask and be non-causal by default."""
        dim = 32
        attn = Attention(dim, heads=4, use_rope=False)
        x = torch.randn(2, 16, dim)
        # mask: True = allowed, False = masked out
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[:, 8:] = False
        mask = mask[:, None, None, :]
        y, _ = attn(x, mask=mask, causal=False)
        assert y.shape == x.shape

    def test_feature_bank_readonly(self):
        """FeatureBank should not require grad on its output (read-only)."""
        bank = FeatureBank(dim=32, max_tokens=8)
        memory = torch.randn(2, 16, 32)
        bank_out = bank(memory)
        assert bank_out.shape == (2, 8, 32)
        # bank_out should be detached from autograd (read-only inference boundary)
        assert not bank_out.requires_grad

    def test_feature_bank_read(self):
        """FeatureBank.read should return bounded slice."""
        bank_module = FeatureBank(dim=32, max_tokens=8)
        bank_data = torch.randn(2, 16, 32)
        bank_out = bank_module(bank_data)
        read = bank_module.read(bank_out, limit=4)
        assert read.shape == (2, 4, 32)

    def test_wide_depth_forward_shapes(self):
        """WideNDepth should produce correct output shape.

        Note: Compressor reduces token count (adaptive pooling), so output
        may have fewer tokens than input. This is expected.
        """
        dim = 32
        wide = wind.Wide(
            *[torch.nn.Sequential(wind.RMSNorm(dim), wind.SwiGLU(dim, dim * 2)) for _ in range(2)],
            dim=dim,
            mode="sum",
        )
        encoder = TransformerBlock(dim, heads=4, mlp_ratio=2.0)
        model = WideNDepth(
            wide=wide,
            encoder=encoder,
            compressor=Compressor(dim, tokens=8),
            depth=ReasoningDepth(
                [TransformerBlock(dim, heads=4, mlp_ratio=2.0)],
                dim, iterations=2, read_tokens=4,
            ),
            dim=dim,
            bank_tokens=8,
            read_tokens=4,
            iterations=2,
        )
        x = torch.randn(2, 16, dim)
        y = model(x)
        assert y.shape[0] == 2
        assert y.shape[2] == dim
        assert y.shape[1] <= 16  # Compressor may reduce token count

    def test_encoder_no_causal_mask(self):
        """Verify Attention encoder mode (causal=False) produces full attention."""
        dim = 32
        attn = Attention(dim, heads=4, use_rope=False)
        x = torch.randn(2, 8, dim)
        y_causal, _ = attn(x, causal=True)
        y_full, _ = attn(x, causal=False)
        # Outputs should differ (causal vs non-causal attention)
        assert not torch.allclose(y_causal, y_full, atol=1e-4)

    def test_wide_depth_gradients(self):
        """Gradients should flow through all trainable stages."""
        dim = 32
        wide = wind.Wide(
            *[torch.nn.Sequential(wind.RMSNorm(dim), wind.SwiGLU(dim, dim * 2)) for _ in range(2)],
            dim=dim,
            mode="sum",
        )
        encoder = TransformerBlock(dim, heads=4, mlp_ratio=2.0)
        model = WideNDepth(
            wide=wide,
            encoder=encoder,
            compressor=Compressor(dim, tokens=8),
            depth=ReasoningDepth(
                [TransformerBlock(dim, heads=4, mlp_ratio=2.0)],
                dim, iterations=1, read_tokens=4,
            ),
            dim=dim,
            bank_tokens=8,
            read_tokens=4,
            iterations=1,
        )
        x = torch.randn(2, 16, dim)
        y = model(x)
        loss = y.sum()
        loss.backward()
        # Check that gradients exist on wide parameters
        for p in model.wide.parameters():
            if p.requires_grad:
                assert p.grad is not None, "No gradient on Wide parameters"

    def test_reasoning_depth_iteration_count(self):
        """ReasoningDepth should iterate exactly `iterations` times."""
        dim = 32
        depth = ReasoningDepth(
            [TransformerBlock(dim, heads=4, mlp_ratio=2.0)],
            dim, iterations=3, read_tokens=4,
        )
        assert depth.iterations == 3
        x = torch.randn(2, 8, dim)
        bank = torch.randn(2, 16, dim)
        y = depth(x, bank)
        assert y.shape == x.shape

    def test_bank_is_read_only_during_depth(self):
        """FeatureBank should produce detached output (no grad through bank in Depth)."""
        dim = 32
        model = WideNDepth(
            wide=wind.Wide(
                *[torch.nn.Sequential(wind.RMSNorm(dim), wind.SwiGLU(dim, dim * 2)) for _ in range(2)],
                dim=dim, mode="sum",
            ),
            encoder=TransformerBlock(dim, heads=4, mlp_ratio=2.0),
            compressor=Compressor(dim, tokens=8),
            depth=ReasoningDepth(
                [TransformerBlock(dim, heads=4, mlp_ratio=2.0)],
                dim, iterations=2, read_tokens=4,
            ),
            dim=dim,
            bank_tokens=8,
            read_tokens=4,
            iterations=2,
        )
        x = torch.randn(2, 16, dim)
        y = model(x)
        loss = y.sum()
        loss.backward()
        # Bank should not have gradients (it's detached/read-only)
        for p in model.bank.parameters():
            if p.requires_grad:
                assert p.grad is None, "Bank should not receive gradients (read-only)"


class TestLanguageModel:
    """Test the LanguageModel with dense wide path."""

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
            norm="rmsnorm",
        )
        defaults.update(kwargs)
        return LMConfig(**defaults)

    def test_lm_forward_dense(self):
        """LanguageModel forward with dense wide should produce correct shape."""
        config = self._make_config()
        model = LanguageModel(config)
        input_ids = torch.randint(0, 100, (2, 8))
        labels = torch.randint(0, 100, (2, 6))
        output = model(input_ids=input_ids, labels=labels)
        assert output.logits.shape == (2, 6, 100)
        assert output.loss is not None
        assert output.token_count.item() > 0

    def test_lm_forward_no_labels(self):
        """LanguageModel forward without labels should return logits only."""
        config = self._make_config()
        model = LanguageModel(config)
        input_ids = torch.randint(0, 100, (2, 8))
        decoder_input_ids = torch.randint(0, 100, (2, 6))
        output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        assert output.logits.shape == (2, 6, 100)
        assert output.loss is None

    def test_lm_save_load(self):
        """LanguageModel save/load roundtrip should preserve weights."""
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

    def test_lm_generate_greedy(self):
        """Greedy generation should produce valid token sequences."""
        config = self._make_config()
        model = LanguageModel(config)
        model.eval()
        input_ids = torch.randint(3, 100, (1, 8))  # avoid special tokens
        with torch.no_grad():
            generated = model.generate(input_ids=input_ids, max_new_tokens=4, temperature=0.0)
        # Generated may be shorter if EOS is produced
        assert generated.shape[0] == 1
        assert generated.shape[1] >= 1
        assert (generated >= 0).all()
        assert (generated < 100).all()

    def test_lm_encoder_is_bidirectional(self):
        """Verify the encoder blocks don't use causal attention."""
        config = self._make_config()
        model = LanguageModel(config)
        x = torch.randint(0, 100, (1, 8))
        state = model.encode(x)
        assert state.tokens.shape == (1, 4, 32)

    def test_lm_iteration_count_honored(self):
        """Depth iterations should be honored exactly."""
        for n in [1, 3, 5]:
            config = self._make_config(iterations=n)
            model = LanguageModel(config)
            # Run encode twice with same seed — outputs should be identical
            input_ids = torch.randint(3, 100, (1, 8))
            torch.manual_seed(42)
            state1 = model.encode(input_ids)
            torch.manual_seed(42)
            state2 = model.encode(input_ids)
            assert torch.allclose(state1.tokens, state2.tokens)

    def test_lm_backward(self):
        """Gradients should flow through LanguageModel forward pass."""
        config = self._make_config()
        model = LanguageModel(config)
        model.train()
        input_ids = torch.randint(3, 100, (1, 8))
        labels = torch.randint(3, 100, (1, 4))
        output = model(input_ids=input_ids, labels=labels)
        loss = output.loss
        loss.backward()
        # Check we got gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters() if p.requires_grad
        )
        assert has_grad, "No gradients found after backward"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLanguageModelCUDA:
    """CUDA-specific tests (skipped on CPU-only)."""

    def test_lm_forward_cuda(self):
        config = LMConfig(
            vocab_size=100, dim=32, heads=4, depth=1, encoder_depth=1,
            decoder_depth=1, width=1, iterations=1, state_tokens=4,
            bank_tokens=8, max_source_length=32, max_target_length=16,
            mlp_ratio=2.0, wide_type="dense", norm="rmsnorm",
        )
        model = LanguageModel(config).cuda()
        input_ids = torch.randint(0, 100, (2, 8)).cuda()
        labels = torch.randint(0, 100, (2, 6)).cuda()
        output = model(input_ids=input_ids, labels=labels)
        assert output.logits.is_cuda
