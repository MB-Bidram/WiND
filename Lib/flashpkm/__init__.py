"""FlashPKM: Optimized Triton kernels and implementations for Product-Key Memory retrieval.

This library provides fused Triton kernels and complete PKM implementations
for high-performance associative memory operations. It is designed as a
standalone, clean dependency that can be used by WiND without risking
corruption from library updates.

Key Optimizations:
- Fused Triton scoring kernel for factor scoring
- Advanced indexing instead of expand+gather
- Efficient value retrieval
- Automatic fallback to pure PyTorch when Triton/CUDA unavailable

Requirements:
    - torch with CUDA support (for Triton kernels)
    - triton >= 2.1 (optional, falls back to pure PyTorch)
"""

from .kernels import (
    fused_factor_scores,
    hierarchical_topk,
    fused_value_aggregate_f2,
)
from .pkm import FactorizedPKM
from .pkm_wide import PKMWide
from .pkm_utils import QueryEncoder, _resolve_norm, _resolve_activation
from .modules import WindModule, RMSNorm, apply_mode
from .execution import PKMInferenceConfig, PKMInferenceEngine

__all__ = [
    # Kernels
    "fused_factor_scores",
    "hierarchical_topk",
    "fused_value_aggregate_f2",

    # Models
    "FactorizedPKM",
    "PKMWide",
    "QueryEncoder",

    # Utilities
    "_resolve_norm",
    "_resolve_activation",
    "apply_mode",

    # Base classes
    "WindModule",
    "RMSNorm",

    # Execution
    "PKMInferenceConfig",
    "PKMInferenceEngine",
]

__version__ = "1.2.0"