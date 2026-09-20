"""Base classes re-exported for backward compatibility.

All base classes (WindModule, RMSNorm, apply_mode) now come from wind.engine
to avoid code duplication. QueryEncoder and the norm/activation resolvers
are re-exported from pkm_utils.
"""

from __future__ import annotations

from winc.modules import WindModule, RMSNorm, StageMode, apply_mode
from .pkm_utils import QueryEncoder, _resolve_norm, _resolve_activation

__all__ = [
    "WindModule",
    "RMSNorm",
    "apply_mode",
    "StageMode",
    "QueryEncoder",
    "_resolve_norm",
    "_resolve_activation",
]
