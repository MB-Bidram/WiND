"""WiND-specific profiling hooks for architecture components.

Hooks are registered with the profiler at trace entry time to avoid
circular imports during package initialization.

When profiling is not active, there is zero overhead — these hooks
are only invoked inside a ``wind.profile(...)`` context.
"""

from __future__ import annotations

from typing import Any

import torch

from ..profiling import register_hook, metric

_HOOKS_REGISTERED = False


def _register_wind_hooks() -> None:
    """Register all built-in WiND hooks (idempotent)."""
    global _HOOKS_REGISTERED
    if _HOOKS_REGISTERED:
        return

    @register_hook
    def wind_component_hook(profiler, action: str) -> None:
        """Record WiND architecture metadata at profiler start/stop."""
        if action == "start":
            profiler._wind_info = {
                "version": "2.3.0",
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            }
        elif action == "stop":
            if hasattr(profiler, "_wind_info"):
                profiler._metrics_buffer.update(profiler._wind_info)

    # ---- Component invocation metrics ----

    def _param_dtype_stats(profiler) -> dict[str, Any]:
        """Count parameters by dtype across the model."""
        if not hasattr(profiler, "_arch_profiler") or profiler._arch_profiler is None:
            return {}
        if not hasattr(profiler._arch_profiler, "_layer_stats"):
            return {}
        dtype_counts: dict[str, int] = {"float32": 0, "float16": 0, "bfloat16": 0, "other": 0}
        for stats in profiler._arch_profiler._layer_stats.values():
            # param_count is already computed; we need dtype info from the model
            pass
        # Try to get dtype info from the model itself
        model = getattr(profiler._arch_profiler, "_model", None)
        if model is not None:
            for p in model.parameters():
                dtype_str = str(p.dtype).replace("torch.", "")
                if dtype_str in dtype_counts:
                    dtype_counts[dtype_str] += p.numel()
                else:
                    dtype_counts["other"] += p.numel()
        return dtype_counts

    def _component_stats(profiler) -> dict[str, int]:
        """Count WiND-specific module invocations from the trace.

        Each invocation of a named component is counted, giving per-component
        kernel-level call counts.
        """
        if not hasattr(profiler, "_trace") or profiler._trace is None:
            return {}
        component_names = {
            "Wide", "WideStack", "Depth", "ReasoningDepth", "Retrieval",
            "FeatureBank", "Compressor", "MLA", "NSA", "RMSNorm", "SwiGLU",
            "Attention", "TransformerBlock", "LearnedQueryCompressor",
            "FeatureBank", "AdaptiveFeatureBank", "EncoderDecoder", "WideNDepth",
        }
        counts = {name: 0 for name in component_names}
        try:
            for event in profiler._trace.events():
                for name in component_names:
                    if name in event.name:
                        counts[name] = counts.get(name, 0) + 1
        except Exception:
            pass
        return counts

    def _kernel_count_by_type(profiler) -> dict[str, int]:
        """Count kernels by category from the profiler trace."""
        if not hasattr(profiler, "_trace") or profiler._trace is None:
            return {}
        counts = {
            "total_kernels": 0,
            "matmul_gemm": 0,
            "elementwise": 0,
            "reduction": 0,
            "softmax": 0,
            "norm": 0,
            "embedding": 0,
            "topk": 0,
            "gather": 0,
            "sort": 0,
            "memory_copy": 0,
        }
        try:
            for event in profiler._trace.events():
                name_lower = event.name.lower()
                counts["total_kernels"] += 1
                if "mm(" in name_lower or "bmm" in name_lower or "gemm" in name_lower or "matmul" in name_lower:
                    counts["matmul_gemm"] += 1
                elif "softmax" in name_lower:
                    counts["softmax"] += 1
                    counts["reduction"] += 1
                elif "rms_norm" in name_lower or "layer_norm" in name_lower or "normalize" in name_lower:
                    counts["norm"] += 1
                elif "embedding" in name_lower or "gather" in name_lower or "index" in name_lower:
                    counts["embedding"] += 1
                    if "gather" in name_lower or "index" in name_lower:
                        counts["gather"] += 1
                elif "topk" in name_lower:
                    counts["topk"] += 1
                elif "sort" in name_lower:
                    counts["sort"] += 1
                elif "copy" in name_lower or "Memcpy" in name_lower:
                    counts["memory_copy"] += 1
                elif "add" in name_lower or "mul" in name_lower or "sub" in name_lower or "div" in name_lower:
                    counts["elementwise"] += 1
                elif "sum" in name_lower or "max" in name_lower or "min" in name_lower or "mean" in name_lower:
                    counts["reduction"] += 1
        except Exception:
            pass
        return counts

    def _attention_stats(profiler) -> dict[str, int]:
        """Count attention-related kernel invocations."""
        if not hasattr(profiler, "_trace") or profiler._trace is None:
            return {}
        counts = {"sdpa": 0, "softmax": 0, "matmul": 0}
        try:
            for event in profiler._trace.events():
                name = event.name.lower()
                if "sdpa" in name or "scaled_dot" in name:
                    counts["sdpa"] += 1
                if "softmax" in name:
                    counts["softmax"] += 1
                if "matmul" in name or "bmm" in name or "mm(" in name:
                    counts["matmul"] += 1
        except Exception:
            pass
        return counts

    def _memory_delta(profiler) -> float:
        """Return the memory delta between profiler start and stop."""
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated() / 1_048_576
            baseline = getattr(profiler, "_wind_mem_start", current)
            return round(current - baseline, 3)
        return 0.0

    def _layer_invocation_counts(profiler) -> dict[str, int]:
        """Count how many times each named layer was invoked during profiling.

        Uses the ArchitectureProfiler layer_stats if available.
        """
        if not hasattr(profiler, "_arch_profiler") or profiler._arch_profiler is None:
            return {}
        if not hasattr(profiler._arch_profiler, "_layer_stats"):
            return {}
        return {
            name: stats.calls
            for name, stats in profiler._arch_profiler._layer_stats.items()
            if stats.calls > 0
        }

    metric("param_dtypes", _param_dtype_stats)
    metric("wind_components", _component_stats)
    metric("kernel_types", _kernel_count_by_type)
    metric("attention_ops", _attention_stats)
    metric("memory_delta_mb", _memory_delta)
    metric("layer_invocations", _layer_invocation_counts)

    _HOOKS_REGISTERED = True


