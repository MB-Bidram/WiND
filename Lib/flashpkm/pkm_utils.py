"""Shared utilities for the PKM Wide implementation.

Re-exports base classes and utilities from wind.engine to avoid duplication.
QueryEncoder is defined here as it is PKM-specific.
"""

from __future__ import annotations

import torch
from torch import nn

from winc.modules import WindModule, RMSNorm
from winc.losses import orthogonality_loss, diversity_loss  # noqa: F401


def _resolve_norm(norm: str | None, dim: int, eps: float = 1e-6) -> nn.Module | None:
    if norm is None or norm == "none":
        return None
    if norm == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    if norm == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"unknown norm: {norm!r} (expected 'none', 'rmsnorm', or 'layernorm')")


def _resolve_activation(name: str | None) -> nn.Module:
    if name is None or name == "identity" or name == "none":
        return nn.Identity()
    table: dict[str, nn.Module] = {
        "silu": nn.SiLU(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "gelu_tanh": nn.GELU(approximate="tanh"),
    }
    if name not in table:
        raise ValueError(f"unknown activation: {name!r}")
    return table[name]


class QueryEncoder(WindModule):
    """Configurable MLP encoder mapping input representations to query space.

    PKM is a memory system, not a universal feature extractor.  This encoder
    is intentionally small and configurable: it learns how to map incoming
    representations into a retrieval-friendly query space.
    """

    def __init__(
        self,
        input_dim: int,
        query_dim: int,
        hidden_dim: int | None = None,
        depth: int = 1,
        activation: str = "silu",
        norm: str | None = None,
        dropout: float = 0.0,
        residual: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        if input_dim < 1 or query_dim < 1:
            raise ValueError("input_dim and query_dim must be positive")
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if hidden_dim is not None and hidden_dim < 1:
            raise ValueError("hidden_dim must be positive when provided")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        hidden_dim = hidden_dim or input_dim
        self.input_dim = input_dim
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.residual = residual and (input_dim == query_dim)

        act = _resolve_activation(activation)
        use_dropout = dropout > 0.0

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for i in range(depth):
            out_dim = query_dim if i == depth - 1 else hidden_dim
            lin = nn.Linear(prev_dim, out_dim, bias=bias)

            if i < depth - 1:
                parts: list[nn.Module] = [lin]
                n = _resolve_norm(norm, out_dim)
                if n is not None:
                    parts.append(n)
                parts.append(act)
                if use_dropout:
                    parts.append(nn.Dropout(dropout))
                layers.append(nn.Sequential(*parts))
            else:
                layers.append(lin)

            prev_dim = out_dim

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.input_dim:
            raise ValueError(f"expected final dimension {self.input_dim}, got {x.size(-1)}")

        residual_in = x
        for layer in self.layers:
            x = layer(x)

        if self.residual:
            x = x + residual_in
        return x

    def estimate_flops(self, batch: int = 1, seq: int = 1) -> int:
        flops = 0
        prev = self.input_dim
        for i, layer in enumerate(self.layers):
            out_dim = self.query_dim if i == len(self.layers) - 1 else self.hidden_dim
            flops += 2 * batch * seq * prev * out_dim
            prev = out_dim
        return flops
