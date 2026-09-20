"""Ready-to-use WideNDepth blocks."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .attention import MLA, NSA
from .modules import FeedForward, RMSNorm, WindModule
from ._internal.guard import assert_tensor, GUARD_ENABLED


class TransformerBlock(WindModule):
    """High-performance pre-norm Transformer block used by WideNDepth.

    The public API is intentionally unchanged.

    Execution:
        x -> norm -> attention -> residual
          -> norm -> SwiGLU -> residual

    The implementation avoids unnecessary temporary Python objects and keeps
    the hot path straightforward for PyTorch eager execution and compilers.
    """

    __slots__ = ()

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        attention: str = "mla",
        dropout: float = 0.0,
        block_size: int = 64,
        topk: int = 4,
        norm: str = "layernorm",
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        if norm not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm must be 'layernorm' or 'rmsnorm'")

        if attention not in {"mla", "nsa"}:
            raise ValueError("attention must be 'mla' or 'nsa'")

        # Resolve the normalization implementation once at construction time.
        norm_cls = nn.LayerNorm if norm == "layernorm" else RMSNorm

        self.norm1 = norm_cls(dim)
        self.norm2 = norm_cls(dim)

        if attention == "mla":
            self.attn = MLA(
                dim,
                heads,
                dropout=dropout,
                use_rope=use_rope,
                rope_theta=rope_theta,
                max_seq_len=max_seq_len,
            )
        else:
            self.attn = NSA(
                dim,
                heads,
                block_size=block_size,
                topk=topk,
                dropout=dropout,
                use_rope=use_rope,
                rope_theta=rope_theta,
                max_seq_len=max_seq_len,
            )

        self.ffn = FeedForward(
            dim,
            int(dim * mlp_ratio),
            dropout,
        )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run the transformer block.

        ``kwargs`` is preserved for attention implementations that accept
        cache/generation-specific arguments.
        """

        # Local bindings avoid repeated Python attribute resolution on the
        # hottest path without changing module registration or state_dicts.
        norm1 = self.norm1
        norm2 = self.norm2
        attn = self.attn
        ffn = self.ffn

        # Most training/forward calls carry no attention kwargs. Keeping that
        # path separate avoids constructing/passing an empty kwargs mapping
        # through the attention stack.
        if kwargs:
            x = x + attn(norm1(x), **kwargs)
        else:
            x = x + attn(norm1(x))

        x = x + ffn(norm2(x))
        return x


class SparseBlock(TransformerBlock):
    """Ready transformer block using the NSA sparse attention path."""

    __slots__ = ()

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        block_size: int = 64,
        topk: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__(
            dim=dim,
            heads=heads,
            mlp_ratio=mlp_ratio,
            attention="nsa",
            dropout=dropout,
            block_size=block_size,
            topk=topk,
        )
