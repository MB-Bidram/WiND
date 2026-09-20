"""Attention implementations with portable PyTorch fallbacks."""

from __future__ import annotations

import math
import sys

import torch
from torch import nn
from torch.nn import functional as F

from .cache import CacheView

from .modules import WindModule
from ._internal.guard import (
    assert_tensor,
    assert_shape,
    assert_dtype,
    assert_device,
    assert_finite,
    GUARD_ENABLED,
)


class RotaryEmbedding(WindModule):
    """LLaMA-style rotary position embedding for query/key tensors.

    Accepts ``[batch, heads, sequence, head_dim]`` and rotates pairs in the
    final dimension. Odd head dimensions are rejected because pairing would be
    ambiguous. The frequency table is precomputed once in ``__init__`` as a
    non-persistent float32 buffer and only sliced in ``forward`` — it is never
    reassigned or grown, which keeps CUDA-Graph and ``torch.compile`` happy.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        if head_dim < 2 or head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if theta <= 0:
            raise ValueError("theta must be positive")
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        self.head_dim, self.theta = head_dim, float(theta)
        self.max_seq_len = max_seq_len

        # Precompute the full frequency table once, in float32, outside any
        # autocast region so bf16 rounding does not degrade long-range
        # positional accuracy.
        with torch.autocast(device_type="cuda", enabled=False):
            inv = 1.0 / (self.theta ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
            ))
            positions = torch.arange(max_seq_len, dtype=torch.float32)
            angles = torch.outer(positions, inv)
            cos = angles.cos()
            sin = angles.sin()

        # Register as non-persistent buffers: they are part of the model's
        # state tracking but should not be saved in the checkpoint, and they
        # must not be overwritten by a compiled-graph replay.
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _values(self, length: int, offset: int, dtype: torch.dtype
                ) -> tuple[torch.Tensor, torch.Tensor]:
        end = offset + length
        if end > self.max_seq_len:
            raise ValueError(
                f"sequence length {end} exceeds precomputed max_seq_len {self.max_seq_len}"
            )
        return (
            self.cos_cached[offset:end].to(dtype=dtype),
            self.sin_cached[offset:end].to(dtype=dtype),
        )

    def forward(self, x: torch.Tensor, *, offset: int = 0) -> torch.Tensor:
        if x.ndim != 4 or x.size(-1) != self.head_dim:
            raise ValueError(f"expected [batch, heads, sequence, {self.head_dim}]")
        end = offset + x.size(-2)
        if end > self.max_seq_len:
            raise ValueError(
                f"sequence length {end} exceeds precomputed max_seq_len {self.max_seq_len}"
            )
        cos = self.cos_cached[offset:end].to(dtype=x.dtype)
        sin = self.sin_cached[offset:end].to(dtype=x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def clear_cache(self) -> None:
        """No-op retained for backward compatibility.

        The cache is now a static precomputed buffer and requires no clearing.
        """
        pass


# Short, discoverable alias for users familiar with the usual name.
RoPE = RotaryEmbedding


class MLA(WindModule):
    """Multi-head latent attention.

    A low-rank latent KV projection reduces the cache width while retaining a
    standard scaled-dot-product execution path (and therefore works on CPU).
    Input/output tensors use ``[batch, sequence, dim]``.
    """

    def __init__(self, dim: int, heads: int = 8, latent_dim: int | None = None,
                 dropout: float = 0.0, causal: bool = True, chunk_size: int | None = None,
                 *, use_rope: bool = False, rope_theta: float = 10000.0,
                 max_seq_len: int = 2048):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.latent_dim = latent_dim or max(self.head_dim, dim // 4)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.latent_kv = nn.Linear(dim, self.latent_dim * 2, bias=False)
        self.k_up = nn.Linear(self.latent_dim, dim, bias=False)
        self.v_up = nn.Linear(self.latent_dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        self.causal = causal
        self.chunk_size = chunk_size
        self.rope = RotaryEmbedding(self.head_dim, rope_theta, max_seq_len) if use_rope else None
        # Precompute position index tensors as non-persistent buffers to avoid
        # per-step torch.arange allocations that block kernel fusion and CUDA-Graph.
        self.register_buffer(
            "_position_cache", torch.arange(max_seq_len, dtype=torch.long),
            persistent=False,
        )

    def prepare_cache(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return compressed latent K/V tensors for request-local KV caching.

        The cache stores ``latent_dim`` channels rather than expanded model
        K/V channels. ``forward_cached`` expands only for the current attention
        operation, keeping the persistent cache small.
        """
        if x.ndim != 3 or x.size(-1) != self.dim:
            raise ValueError(f"expected [batch, sequence, {self.dim}]")
        return self.latent_kv(x).chunk(2, dim=-1)

    def forward_cached(self, x: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor],
                       *, position_offset: int = 0, return_cache: bool = False):
        """Attend to a prior latent cache plus ``x`` and optionally extend it.

        This API is intended for autoregressive serving. For training, call the
        normal ``forward`` path so SDPA can use its best full-sequence kernel.
        """
        if x.ndim != 3 or x.size(-1) != self.dim:
            raise ValueError(f"expected [batch, sequence, {self.dim}]")
        old_k, old_v = cache
        if old_k.ndim != 3 or old_v.shape != old_k.shape or old_k.size(-1) != self.latent_dim:
            raise ValueError("cache must contain [batch, sequence, latent_dim] tensors")
        current = self.prepare_cache(x)
        latent_k = torch.cat((old_k, current[0]), dim=1)
        latent_v = torch.cat((old_v, current[1]), dim=1)
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_up(latent_k).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_up(latent_v).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q = self.rope(q, offset=position_offset)
            k = self.rope(k, offset=0)
        query_positions = self._position_cache[position_offset:position_offset + n][:, None]
        key_positions = self._position_cache[:k.size(-2)][None, :]
        allowed = key_positions <= query_positions
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        result = self.out_proj(y.transpose(1, 2).reshape(b, n, self.dim))
        if GUARD_ENABLED:
            assert_finite(result, "mla_output", self)
        return (result, (latent_k, latent_v)) if return_cache else result

    def forward(self, x: torch.Tensor, *, mask: torch.Tensor | None = None,
                position_offset: int = 0) -> torch.Tensor:
        assert_tensor(x, "input", shape=(None, None, self.dim), context=self)
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        latent = self.latent_kv(x)
        lk, lv = latent.chunk(2, dim=-1)
        k = self.k_up(lk).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_up(lv).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, offset=position_offset), self.rope(k, offset=position_offset)
        if self.chunk_size and n > self.chunk_size:
            # Build a single causal mask using cached position indices, then run
            # one fused SDPA call instead of Python-looping over chunks.
            pos = self._position_cache[:n][None, None, :, None]
            causal_mask = self._position_cache[:n][None, None, None, :] <= pos
            chunk_mask = causal_mask if mask is None else (mask[:, None, :, :] & causal_mask)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=chunk_mask, dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=self.causal and mask is None)
        return self.out_proj(y.transpose(1, 2).reshape(b, n, self.dim))


class NSA(WindModule):
    """Native Sparse Attention-style block attention.

    The compressed, selected, and local paths are represented by a portable
    block selector. For short inputs or unsupported kernels it falls back to
    dense attention, keeping behavior predictable while avoiding fragile
    custom CUDA dependencies.
    """

    def __init__(self, dim: int, heads: int = 8, block_size: int = 64,
                 topk: int = 4, window_size: int = 128, dropout: float = 0.0,
                 causal: bool = True, *, use_rope: bool = False,
                 rope_theta: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if block_size < 1 or topk < 1 or window_size < 1:
            raise ValueError("block_size, topk, and window_size must be positive")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.block_size, self.topk, self.window_size = block_size, topk, window_size
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout, self.causal = dropout, causal
        self.rope = RotaryEmbedding(self.head_dim, rope_theta, max_seq_len) if use_rope else None
        # Precompute position index tensors as non-persistent buffers to avoid
        # per-step torch.arange allocations that block kernel fusion and CUDA-Graph.
        self.register_buffer(
            "_position_cache", torch.arange(max_seq_len, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_block_index_cache",
            torch.arange(max_seq_len // block_size + 1, dtype=torch.long) * block_size,
            persistent=False,
        )
        self.register_buffer(
            "_local_offset_cache",
            torch.arange(max(window_size // block_size, 1), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_block_size_cache",
            torch.arange(block_size, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, *, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        # Dense SDPA is faster and more stable for short sequences. The block
        # mask is only materialized for long sequences, where sparsity helps.
        if n <= self.block_size * max(self.topk, 1) or not x.is_cuda:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal and mask is None,
            )
        else:
            y = self._sparse_sdpa(q, k, v, mask)
        return self.out_proj(y.transpose(1, 2).reshape(b, n, self.dim))

    def _sparse_sdpa(self, q, k, v, mask=None):
        """Select key blocks from block summaries, then attend only to tokens there.

        This avoids the dense ``[batch, heads, query, key]`` score and mask
        tensors. The summary is a differentiable mean-key proxy; selected
        indices are intentionally discrete, as in block-sparse routing.
        """
        n = q.size(-2)
        blocks = (n + self.block_size - 1) // self.block_size
        padded = blocks * self.block_size
        pad = padded - n
        if pad:
            k_pad = F.pad(k, (0, 0, 0, pad))
            v_pad = F.pad(v, (0, 0, 0, pad))
        else:
            k_pad, v_pad = k, v
        k_blocks = k_pad.reshape(*k.shape[:2], blocks, self.block_size, self.head_dim)
        v_blocks = v_pad.reshape(*v.shape[:2], blocks, self.block_size, self.head_dim)
        block_keys = k_blocks.mean(dim=-2)
        summary = torch.matmul(q, block_keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.causal:
            q_positions = self._position_cache[:n].view(1, 1, n, 1)
            block_start = self._block_index_cache[:blocks].view(1, 1, 1, blocks)
            summary = summary.masked_fill(block_start > q_positions, -torch.inf)
        keep = summary.topk(min(max(self.topk, 1), blocks), dim=-1).indices
        local_count = max(1, math.ceil(self.window_size / self.block_size))
        q_blocks = (self._position_cache[:n] // self.block_size).view(1, 1, n, 1)
        offsets = self._local_offset_cache[:local_count].view(1, 1, 1, local_count)
        local = (q_blocks - offsets).clamp_min(0)
        keep = torch.cat((keep, local.expand(q.size(0), q.size(1), -1, -1)), dim=-1)
        token_index = keep.unsqueeze(-1) * self.block_size + self._block_size_cache
        token_index = token_index.reshape(*token_index.shape[:-2], -1)
        valid = token_index < n
        token_index = token_index.clamp_max(n - 1)
        gather_index = token_index.unsqueeze(-1).expand(-1, q.size(1), -1, -1, self.head_dim)
        selected_k = k.unsqueeze(2).expand(-1, -1, n, -1, -1).gather(3, gather_index)
        selected_v = v.unsqueeze(2).expand(-1, -1, n, -1, -1).gather(3, gather_index)
        scores = (q.unsqueeze(3) * selected_k).sum(-1) / math.sqrt(self.head_dim)
        allowed = valid
        if self.causal:
            q_positions = self._position_cache[:n].view(1, 1, n, 1)
            allowed = allowed & (token_index <= q_positions)
        if mask is not None:
            key_mask = mask
            if key_mask.ndim == 2:
                key_mask = key_mask[:, None, None, :]
            elif key_mask.ndim == 3:
                key_mask = key_mask[:, None, :, :]
            elif key_mask.ndim != 4:
                raise ValueError("mask must have 2, 3, or 4 dimensions")
            key_mask = key_mask.expand(q.size(0), q.size(1), n, -1)
            allowed = allowed & key_mask.gather(-1, token_index)
        scores = scores.masked_fill(~allowed, -torch.inf)
        # Every causal query has at least one selected key; protect malformed masks.
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        result = (weights.unsqueeze(-1) * selected_v).sum(dim=-2)
        return result.masked_fill(~allowed.any(dim=-1, keepdim=True), 0)


class Attention(WindModule):
    """Standard multi-head attention with SDPA and optional KV caching.

    This is the canonical attention implementation for decoder/encoder blocks.
    Supports self-attention, cross-attention, causal masking, and request-local
    K/V caching for autoregressive generation.

    Input/output tensors use ``[batch, sequence, dim]``.
    """

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0,
                 use_rope: bool = False, rope_theta: float = 10000.0,
                 max_seq_len: int = 2048):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        # Fused QKV projection reduces kernel launches vs separate q/k/v linears.
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        # Separate KV projection for cross-attention — avoids computing
        # a query projection on the context tensor (wasted 1/3 of the matmul).
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim, rope_theta, max_seq_len) if use_rope else None

    def split(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), x.size(1), self.heads, self.head_dim).transpose(1, 2)

    def project_q(self, x: torch.Tensor) -> torch.Tensor:
        """Project query alone from a fused QKV call."""
        return self.split(self.qkv(x).chunk(3, dim=-1)[0])

    def project_kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project key and value from the dedicated KV projection."""
        k, v = self.kv(x).chunk(2, dim=-1)
        return self.split(k), self.split(v)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None,
                *, mask: torch.Tensor | None = None, causal: bool = False,
                cache: tuple[torch.Tensor, torch.Tensor] | CacheView | None = None,
                use_cache: bool = False, static: bool = False,
                position_offset: int = 0) -> tuple[torch.Tensor, ...]:
        cache_view = cache if isinstance(cache, CacheView) else None
        cached_kv = cache_view.get() if cache_view is not None else cache
        if static and cached_kv is not None:
            k, v = cached_kv
            q = self.split(self.qkv(x).chunk(3, dim=-1)[0])
        else:
            # Fused QKV: query from x, key/value from context (cross-attn) or x (self-attn).
            q_part, k_part, v_part = self.qkv(x).chunk(3, dim=-1)
            q = self.split(q_part)
            if context is not None:
                # Cross-attention: project only k/v from context, skipping
                # the wasted query projection on the context tensor.
                k, v = self.project_kv(context)
            else:
                k = self.split(k_part)
                v = self.split(v_part)
                if self.rope is not None:
                    q = self.rope(q, offset=position_offset)
                    k = self.rope(k, offset=position_offset)
            if cache_view is not None:
                cache_view.append(k, v)
                k, v = cache_view.get()
            elif cached_kv is not None:
                k = torch.cat((cached_kv[0], k), dim=-2)
                v = torch.cat((cached_kv[1], v), dim=-2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal and cached_kv is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = self.out(y.transpose(1, 2).reshape(x.shape))
        if GUARD_ENABLED:
            assert_finite(y, "attention_output", self)
        return y, (k, v) if use_cache else None
