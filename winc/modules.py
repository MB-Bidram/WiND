"""Core width/depth building blocks.

The core layer deliberately has no convenience syntax; wrappers in ``wind.wrap``
provide the small public API exposed at the WiND package level.
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ._internal.guard import (
    assert_tensor, assert_dtype, assert_finite, assert_shape,
    GUARD_ENABLED,
)


_RMS_NORM = getattr(F, "rms_norm", None)


class WindModule(nn.Module):
    """Base class for Wind modules.

    Subclasses can override :meth:`estimate_flops` to expose lightweight
    hardware-aware planning information without coupling execution to a
    particular device.
    """

    def estimate_flops(self, *args, **kwargs) -> int | None:
        return None

    def prepare_for_inference(self, dtype: torch.dtype | None = None) -> "WindModule":
        if dtype is not None:
            self.to(dtype=dtype)
        return self


class Wide(WindModule):
    """Run branches in parallel and merge them.

    ``mode="sum"`` is parameter-free and preserves the input width. ``mode=
    "concat"`` concatenates branch outputs and projects back to ``dim``.
    """

    def __init__(self, *branches: nn.Module, dim: int | None = None, mode: str = "sum"):
        super().__init__()
        if not branches:
            raise ValueError("Wide requires at least one branch")
        if mode not in {"sum", "concat"}:
            raise ValueError("mode must be 'sum' or 'concat'")
        self.branches = nn.ModuleList(branches)
        self.mode = mode
        if mode == "concat" and dim is None:
            raise ValueError("dim is required when mode='concat'")
        self.proj = nn.LazyLinear(dim) if mode == "concat" and dim is not None else None

    def forward(self, x: torch.Tensor, *, return_branches: bool = False):
        branches = self.branches
        assert_tensor(x, "Wide.input", context=self)

        # The single-branch case does not need any merge operation.
        if len(branches) == 1:
            y = branches[0](x)
            if GUARD_ENABLED:
                assert_finite(y, "Wide.branch_output", self)
            if self.mode == "sum":
                return (y, y.unsqueeze(0)) if return_branches else y

            merged = self.proj(y) if self.proj is not None else y
            if GUARD_ENABLED:
                assert_finite(merged, "Wide.merged", self)
            return (merged, y.unsqueeze(0)) if return_branches else merged

        outputs = [branch(x) for branch in branches]

        if self.mode == "sum":
            shape = outputs[0].shape
            if any(y.shape != shape for y in outputs[1:]):
                raise ValueError("Wide(sum) branches must return equal shapes")

            # Stack once and sum; under torch.compile this fuses into a single
            # kernel. The stacked tensor is reused when branch outputs are requested.
            stacked = torch.stack(outputs, dim=0)
            merged = stacked.sum(dim=0)
            return (merged, stacked) if return_branches else merged

        y = torch.cat(outputs, dim=-1)
        merged = self.proj(y) if self.proj is not None else y

        if return_branches:
            return merged, torch.stack(outputs, dim=0)
        return merged

    def forward_with_branches(self, x: torch.Tensor):
        """Return the merged output and branch outputs for regularization."""

        return self.forward(x, return_branches=True)


class WideStack(WindModule):
    """Compose multiple Wide stages sequentially.

    Each stage may use a different ordinary ``nn.Module`` architecture. This
    makes repeated Wide passes explicit while preserving the tensor contract.
    """

    def __init__(self, *stages: nn.Module):
        super().__init__()
        if not stages:
            raise ValueError("WideStack requires at least one stage")
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor, *, return_branches: bool = False):
        captures = [] if return_branches else None

        for stage in self.stages:
            if return_branches and hasattr(stage, "forward_with_branches"):
                x, branches = stage.forward_with_branches(x)
                captures.append(branches)
            else:
                x = stage(x)

        if not return_branches:
            return x

        if not captures:
            return x, None

        if len(captures) == 1:
            return x, captures[0]

        return x, torch.cat(captures, dim=0)

    def forward_with_branches(self, x: torch.Tensor):
        return self.forward(x, return_branches=True)


class Depth(WindModule):
    """Apply modules sequentially while preserving the tensor contract."""

    def __init__(self, *layers: nn.Module, checkpointing: bool = False):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.checkpointing = checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        checkpointing = self.checkpointing
        training = self.training

        if checkpointing and training and torch.is_grad_enabled():
            for layer in self.layers:
                x = checkpoint(layer, x, use_reentrant=False)
            return x

        for layer in self.layers:
            x = layer(x)
        return x


class RMSNorm(WindModule):
    """Root-mean-square normalization used by modern transformer blocks."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.dim, self.eps = dim, eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.dim:
            raise ValueError(f"expected final dimension {self.dim}")

        weight = self.weight
        # Ensure weight matches input dtype to enable fused kernels.
        if weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)

        if _RMS_NORM is not None and weight.dtype == x.dtype:
            return _RMS_NORM(
                x,
                (self.dim,),
                weight,
                self.eps,
            )

        # Accumulate in fp32 for fp16/bf16 stability, then restore the input dtype.
        variance = x.float().square().mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        y = x * scale * weight
        if GUARD_ENABLED:
            assert_finite(y, "rmsnorm_output", self)
        return y


class SwiGLU(WindModule):
    """Parameter-efficient gated feed-forward projection."""

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0,
                 bias: bool = False):
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        hidden_dim = hidden_dim or 4 * dim
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert_tensor(x, "SwiGLU.input", context=self)
        hidden = F.silu(self.gate(x)) * self.up(x)
        y = self.down(hidden)

        if GUARD_ENABLED:
            assert_finite(y, "SwiGLU.output", self)

        # Avoid the Dropout module/kernel entirely in the overwhelmingly
        # common inference/default-training case of dropout=0.
        if self.dropout.p == 0.0:
            return y
        return self.dropout(y)


class FeedForward(SwiGLU):
    """Gated SwiGLU feed-forward network used by ready blocks."""

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0,
                 *, bias: bool = False):
        super().__init__(dim, hidden_dim, dropout, bias=bias)


def as_module(value: nn.Module | Iterable[nn.Module]) -> nn.Module:
    """Normalize a module or iterable into a core module."""

    if isinstance(value, nn.Module):
        return value
    return Depth(*tuple(value))


class StageMode(WindModule):
    """Apply a reusable transform to any stage (Wide, Compressor, Depth, ...).

    ``mode`` may be a named built-in or a callable ``fn(x)``.
    """

    def __init__(
        self,
        module: nn.Module,
        mode: str | Callable[[torch.Tensor], torch.Tensor] = "identity",
        **kwargs,
    ):
        super().__init__()
        self.module = module
        self.mode = mode
        self.kwargs = kwargs

        if isinstance(mode, str) and mode not in {"identity", "residual", "norm", "gated"}:
            raise ValueError("unknown stage mode")
        if mode == "norm" and "dim" not in kwargs:
            raise ValueError("dim is required when mode='norm'")

        self.gate = nn.Parameter(torch.zeros(1)) if mode == "gated" else None
        self.norm = nn.LayerNorm(kwargs["dim"]) if mode == "norm" and "dim" in kwargs else None

    def forward(self, x: torch.Tensor, **kwargs):
        y = self.module(x, **kwargs)

        if callable(self.mode):
            return self.mode(y)
        if self.mode == "identity":
            return y
        if self.mode == "residual":
            return x + y
        if self.mode == "gated":
            return x + torch.sigmoid(self.gate) * y
        return self.norm(y) if self.norm is not None else y


def apply_mode(module: nn.Module, mode: str | Callable = "identity", **kwargs) -> nn.Module:
    """One small hook for wrapping any WND stage with a built-in/custom mode."""
    return StageMode(module, mode, **kwargs)