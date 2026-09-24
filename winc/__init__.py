"""WiNC: low-level stable computation backend for WiND.

WiNC owns low-level tensor operations, optimized modules and kernels,
backend execution, device and dtype handling, memory/retrieval primitives,
compilation and execution policies, runtime guards, caching, and the
integration boundary for FlashPKM.

WiNC must NOT depend on WiND. The dependency direction is:

    WiNC  <-  WiND

Modules:
    _internal.guard  - standardized error/warning reporting and assertions
    modules          - WindModule, Wide, WideStack, Depth, RMSNorm, SwiGLU, FeedForward, StageMode
    attention        - RotaryEmbedding/RoPE, MLA, NSA, Attention
    blocks           - TransformerBlock, SparseBlock
    architecture     - LearnerQueryCompressor, Compressor, FeatureBank, Retrieval, ReasoningDepth, WideNDepth, EncoderDecoder
    adapters         - LinearAdapter, TextAdapter, ImageAdapter, VideoAdapter, AudioAdapter
    losses           - orthogonality_loss, diversity_loss
    runtime          - HardwareProfile, OptimizedBackend, optimize, _compile_ctx
    logger           - console_logger, json_logger, make_logger
    cache            - GenerationCache, LayerCache, CacheView
    pkm              - FlashPKM bridge (make_pkm_wide, PKMAvailable)
"""

from ._internal.guard import (
    WindError,
    WindWarning,
    WindDeprecationWarning,
    WindAssertError,
    guard,
    assert_tensor,
    assert_finite,
    assert_dtype,
    assert_device,
    assert_shape,
    assert_shape_compatible,
    warn_sync,
    warn_dtype_mismatch,
    warn_device_mismatch,
    set_guard_enabled,
    set_guard_strict,
    assert_amp_compatible,
    assert_no_fp16_master_weights,
    assert_not_in_hot_path,
    assert_gradient_has_values,
    assert_loss_monotonically_decreasing,
    track_compile_start,
    check_compile_timeout,
    track_compile_end,
    reset_compile_tracker,
    GUARD_ENABLED,
    GUARD_STRICT,
)
from .modules import (
    WindModule,
    Wide,
    WideStack,
    Depth,
    RMSNorm,
    SwiGLU,
    FeedForward,
    StageMode,
    apply_mode,
    as_module,
)
from .attention import (
    RotaryEmbedding,
    RoPE,
    MLA,
    NSA,
    Attention,
)
from .blocks import (
    TransformerBlock,
    SparseBlock,
)
from .architecture import (
    LearnedQueryCompressor,
    Compressor,
    FeatureBank,
    AdaptiveFeatureBank,
    Retrieval,
    ReasoningDepth,
    WideNDepth,
    EncoderDecoder,
)
from .adapters import (
    LinearAdapter,
    TextAdapter,
    ImageAdapter,
    VideoAdapter,
    AudioAdapter,
)
from .losses import (
    orthogonality_loss,
    diversity_loss,
)
from .runtime import (
    HardwareProfile,
    OptimizedBackend,
    optimize,
    _compile_ctx,
)
from .logger import (
    Logger,
    console_logger,
    json_logger,
    make_logger,
)
from .cache import (
    GenerationCache,
    LayerCache,
    CacheView,
)
from .pkm import (
    PKMAvailable,
    make_pkm_wide,
    pkm_available,
)
from .smart_compile import (
    compile,
    CompileConfig,
    CompileProgress,
    get_compile_stats,
    reset_compile_stats,
)

__all__ = [
    # guard
    "WindError", "WindWarning", "WindDeprecationWarning", "WindAssertError",
    "guard", "assert_tensor", "assert_finite", "assert_dtype", "assert_device",
    "assert_shape", "assert_shape_compatible", "warn_sync", "warn_dtype_mismatch",
    "warn_device_mismatch", "set_guard_enabled", "set_guard_strict",
    "assert_amp_compatible", "assert_no_fp16_master_weights", "assert_not_in_hot_path",
    "assert_gradient_has_values", "assert_loss_monotonically_decreasing",
    "track_compile_start", "check_compile_timeout", "track_compile_end",
    "reset_compile_tracker", "GUARD_ENABLED", "GUARD_STRICT",
    # modules
    "WindModule", "Wide", "WideStack", "Depth", "RMSNorm", "SwiGLU", "FeedForward",
    "StageMode", "apply_mode", "as_module",
    # attention
    "RotaryEmbedding", "RoPE", "MLA", "NSA", "Attention",
    # blocks
    "TransformerBlock", "SparseBlock",
    # architecture
    "LearnedQueryCompressor", "Compressor", "FeatureBank", "AdaptiveFeatureBank",
    "Retrieval", "ReasoningDepth", "WideNDepth", "EncoderDecoder",
    # adapters
    "LinearAdapter", "TextAdapter", "ImageAdapter", "VideoAdapter", "AudioAdapter",
    # losses
    "orthogonality_loss", "diversity_loss",
    # runtime
    "HardwareProfile", "OptimizedBackend", "optimize", "_compile_ctx",
    # logger
    "Logger", "console_logger", "json_logger", "make_logger",
    # cache
    "GenerationCache", "LayerCache", "CacheView",
    # pkm (FlashPKM bridge)
    "PKMAvailable", "make_pkm_wide", "pkm_available",
    "FactorizedPKM", "PKMWide", "QueryEncoder",
    # smart_compile
    "compile", "CompileConfig", "CompileProgress", "get_compile_stats",
    "reset_compile_stats",
]


def __getattr__(name: str):
    """Lazily expose optional FlashPKM classes without importing them on CPU."""
    if name in {"FactorizedPKM", "PKMWide", "QueryEncoder"}:
        from . import pkm
        return getattr(pkm, name)
    raise AttributeError(name)
