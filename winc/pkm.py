"""FlashPKM bridge for WiNC.

This module provides the stable interface through which WiND accesses
FlashPKM functionality. WiND must never import FlashPKM directly; all
PKM access goes through the functions and classes exported here.

The bridge exposes:
    - PKMAvailable: exception raised when PKM is requested but unavailable
    - pkm_available(): check whether FlashPKM Triton kernels are usable
    - make_pkm_wide(): construct a PKMWide module backed by FlashPKM

Dependency direction:
    WiND -> winc.pkm -> flashpkm (Lib/flashpkm)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .modules import WindModule


class PKMAvailable(RuntimeError):
    """Raised when FlashPKM is requested but unavailable.

    FlashPKM requires PyTorch with optional Triton for CUDA kernels.
    On CPU-only or no-Triton environments, the pure-PyTorch fallback
    path may be available, but fused Triton kernels are not.
    """

    pass


def pkm_available() -> bool:
    """Return True if FlashPKM can be imported and used."""
    try:
        import flashpkm  # noqa: F401
        return True
    except ImportError:
        return False


def make_pkm_wide(
    *,
    dim: int,
    memory_size: int = 4096,
    query_dim: int | None = None,
    value_dim: int | None = None,
    num_factors: int = 2,
    heads: int = 4,
    topk: int = 32,
    topk_per_factor: int | None = None,
    similarity: str = "cosine",
    output_mode: str = "none",
    key_dtype: str | torch.dtype = "float32",
    value_dtype: str | torch.dtype = "float32",
    **kwargs: Any,
) -> nn.Module:
    """Construct a PKMWide module through the FlashPKM bridge.

    This is the only WiNC public path for creating a PKM-backed Wide.
    WiND calls this function instead of importing PKMWide directly.

    Args:
        dim: Model dimension (input/output of the Wide module).
        memory_size: Memory slots per sub-key table.
        query_dim: Query encoding dimension. Defaults to dim // 2.
        value_dim: Value vector dimension. Defaults to dim.
        num_factors: Number of factorized sub-key tables.
        heads: Number of retrieval heads.
        topk: Global top-k candidates retrieved per head.
        topk_per_factor: Per-factor top-k before Cartesian product.
        similarity: "cosine" (recommended) or "dot".
        output_mode: "none", "residual", or "gated".
        key_dtype: Key storage dtype (string or torch.dtype).
        value_dtype: Value storage dtype (string or torch.dtype).
        **kwargs: Additional arguments forwarded to PKMWide.

    Returns:
        A PKMWide nn.Module.

    Raises:
        PKMAvailable: If FlashPKM is not available.
        ValueError: If query_dim is not divisible by heads * num_factors.
    """
    if not pkm_available():
        raise PKMAvailable(
            "FlashPKM is not available. Ensure flashpkm is installed "
            "(pip install -e Lib/flashpkm) and triton>=2.1 is available "
            "for CUDA kernels. For CPU-only environments, use wide_type='dense'."
        )

    from flashpkm import PKMWide

    if not isinstance(key_dtype, torch.dtype):
        _dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
        }
        if key_dtype not in _dtype_map:
            raise ValueError(f"unknown key_dtype: {key_dtype!r}")
        key_dtype = _dtype_map[key_dtype]

    if not isinstance(value_dtype, torch.dtype):
        _dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if value_dtype not in _dtype_map:
            raise ValueError(f"unknown value_dtype: {value_dtype!r}")
        value_dtype = _dtype_map[value_dtype]

    if query_dim is None:
        query_dim = dim // 2

    if value_dim is None:
        value_dim = dim

    head_dim = query_dim // (heads * num_factors)
    if query_dim != head_dim * heads * num_factors:
        remainder = query_dim % (heads * num_factors)
        raise ValueError(
            f"query_dim ({query_dim}) must be divisible by "
            f"heads ({heads}) * num_factors ({num_factors}). "
            f"Remainder: {remainder}. Try adjusting dim, heads, or num_factors."
        )

    return PKMWide(
        dim=dim,
        memory_size=memory_size,
        query_dim=query_dim,
        value_dim=value_dim,
        num_factors=num_factors,
        heads=heads,
        topk=topk,
        topk_per_factor=topk_per_factor,
        similarity=similarity,
        output_mode=output_mode,
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        **kwargs,
    )


# Convenience re-exports are deliberately lazy.  Importing a dense WiND model
# must not import optional FlashPKM/Triton modules merely to expose names.
_PKM_CLASS_NAMES = {"FactorizedPKM", "PKMWide", "QueryEncoder"}


def __getattr__(name: str):
    if name in _PKM_CLASS_NAMES:
        try:
            import flashpkm
        except ImportError as exc:
            raise AttributeError(
                f"{name} requires the optional FlashPKM dependency; install wind[pkm]"
            ) from exc
        return getattr(flashpkm, name)
    raise AttributeError(name)
