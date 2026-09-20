"""WiND smart compile: auto-optimized torch.compile with progress tracking.

Usage::

    model = wind.compile(model, mode="auto", show_progress=True)

Automatically selects the best compilation mode based on:
- GPU architecture (SMs, VRAM)
- Model size (param count)
- Whether cudagraphs will help (small SM count GPUs)

Tracks recompilations and errors with user-friendly warnings.
"""

from __future__ import annotations

import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn

from ._internal.guard import (
    GUARD_ENABLED,
    WindWarning,
    _warn_cache,
    guard,
)

__all__ = [
    "CompileConfig",
    "compile",
    "CompileProgress",
    "get_compile_stats",
    "reset_compile_stats",
]

_COMPILE_STATS: dict[str, int] = {"recompiles": 0, "errors": 0, "total_compiles": 0}


@dataclass
class CompileConfig:
    """Configuration for wind.compile().

    Attributes:
        mode: "auto", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs",
              "default", or a custom callable returning a mode string.
        dynamic: Whether to use dynamic shapes (False = static for CUDA Graphs).
        fullgraph: Whether to require fullgraph compilation.
        cudagraphs: "auto", "on", or "off". "auto" disables on GPUs with few SMs.
        verbose: Enable torch.compile verbose output.
        show_progress: Show compilation progress bar (perf-friendly: only on first compile).
        backend: torch.compile backend ("inductor" default).
        disable: Skip compilation entirely (returns model unchanged).
    """

    mode: str = "auto"
    dynamic: bool | None = None
    fullgraph: bool = False
    cudagraphs: str = "auto"
    verbose: bool = False
    show_progress: bool = True
    backend: str = "inductor"
    disable: bool = False
    cache_size_limit: int = 64
    fallback_to_eager: bool = False


class CompileProgress:
    """Lightweight progress indicator for torch.compile.

    Uses write-based printing (not tqdm) to avoid importing heavy dependencies
    and to remain compatible with all stdout/stderr redirection. Only prints
    on the first compilation per process to avoid perf overhead.
    """

    _printed: bool = False
    _start_time: float = 0.0
    _last_update: float = 0.0
    _step: int = 0
    _total_estimate: int = 0
    _stream: Any

    def __init__(self, total_estimate: int = 5, stream: Any = None):
        self._total_estimate = total_estimate
        self._stream = stream or sys.stderr
        self._printed = False
        self._step = 0

    def start(self, model_name: str, mode: str) -> None:
        if not GUARD_ENABLED:
            return
        self._start_time = time.perf_counter()
        self._last_update = self._start_time
        name = getattr(model_name, "__class__", model_name)
        class_name = getattr(name, "__name__", str(name))
        msg = f"  [wind.compile] Compiling {class_name} (mode={mode})..."
        print(msg, file=self._stream, flush=True)
        self._printed = True

    def step(self, description: str = "") -> None:
        """Update progress. Only prints every ~0.5s to avoid perf overhead."""
        if not self._printed:
            return
        now = time.perf_counter()
        if now - self._last_update < 0.5:
            return
        self._last_update = now
        self._step += 1
        elapsed = now - self._start_time
        if elapsed > 0.1:
            rate = self._step / elapsed
            eta = (self._total_estimate - self._step) / max(rate, 1e-9)
            msg = f"    ...{self._step}/{self._total_estimate} stages done ({eta:.1f}s eta)"
            print(msg, file=self._stream, flush=True)

    def finish(self, success: bool = True) -> None:
        if not self._printed:
            return
        elapsed = time.perf_counter() - self._start_time
        status = "done" if success else "failed"
        msg = f"  [wind.compile] {status} in {elapsed:.1f}s"
        print(msg, file=self._stream, flush=True)
        self._printed = False


_recompilation_warnings: dict[int, int] = {}


def _check_recompilation_count(module: nn.Module) -> None:
    """Check for excessive recompilations and warn."""
    from torch._dynamo import utils as dynamo_utils

    try:
        counters = dynamo_utils.counters
        recompiles = sum(counters.get("recompiles", {}).values())
        guard_hash = id(module)
        prev = _recompilation_warnings.get(guard_hash, 0)

        if recompiles > prev:
            delta = recompiles - prev
            if delta > 5:
                _warn_cache.clear()  # force re-warn
                msg = (
                    f"torch.compile has triggered {delta} additional recompilations "
                    f"(total: {recompiles}). This may indicate: "
                    "1) dynamic shapes in the hot path, 2) tensor.data / .item() calls, "
                    "3) conditional branches on runtime values. "
                    "Use static=True or guard against recompilation."
                )
                guard("WARNING", msg, "WiNDCompile", category=WindWarning)
            _recompilation_warnings[guard_hash] = recompiles
    except Exception:
        pass


def _detect_recompilation_risk(module: nn.Module) -> None:
    """Static analysis to detect common recompilation triggers in a module."""
    risk_items: list[str] = []

    def _check_tensor_ops(name: str, mod: nn.Module) -> None:
        for attr_name in ("forward", "__call__"):
            if hasattr(mod, attr_name):
                target = getattr(mod, attr_name)
                if callable(target):
                    try:
                        src = torch._C._get_source(target)
                        if src and any(
                            pattern in src for pattern in
                            ["torch._assert", "if ", "for ", "while ", "len(", ".item("]
                        ):
                            if ".item(" in src or "_assert" in src:
                                risk_items.append(f"{name}: {attr_name} has potential sync/sync-point")
                            elif "if " in src or "while " in src:
                                risk_items.append(f"{name}: {attr_name} has dynamic control flow")
                    except Exception:
                        pass

    for name, mod in module.named_modules():
        if mod is not module:
            _check_tensor_ops(name, mod)

    if risk_items:
        msg = f"Potential recompilation risk in {len(risk_items)} modules:\n  " + "\n  ".join(risk_items)
        guard("WARNING", msg, "WiNDCompile", category=WindWarning)


def _select_best_mode(config: CompileConfig, model: nn.Module) -> str:
    """Auto-select the optimal torch.compile mode based on hardware + model size."""
    if config.mode != "auto":
        return config.mode

    total_params = sum(p.numel() for p in model.parameters())

    # Determine GPU capability
    sm_count = getattr(torch.cuda, "get_device_properties", lambda x: None)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        sm_count_val = props.multi_processor_count
        total_sm = sum(
            torch.cuda.get_device_properties(i).multi_processor_count
            for i in range(torch.cuda.device_count())
        )
    else:
        sm_count_val = 0
        total_sm = 0

    # Small GPUs (laptops etc.): avoid max-autotune overhead
    if total_sm <= 20:
        return "max-autotune-no-cudagraphs"

    # Medium GPUs: reduce-overhead for batch=1 workloads, max-autotune otherwise
    if total_params < 50_000_000 and total_sm <= 40:
        return "reduce-overhead"

    # Large GPUs with big models: full max-autotune
    if total_sm >= 80 and total_params >= 100_000_000:
        return "max-autotune"

    # Default: no cudagraphs for safety
    return "max-autotune-no-cudagraphs"


def _select_cudagraphs(config: CompileConfig) -> str:
    """Auto-select CUDA graphs setting."""
    if config.cudagraphs != "auto":
        return config.cudagraphs

    if not torch.cuda.is_available():
        return "off"

    props = torch.cuda.get_device_properties(0)
    if props.multi_processor_count <= 20:
        return "off"

    return "on"


def _setup_compile_env(config: CompileConfig) -> dict[str, Any]:
    """Set up environment for compilation.

    Handles Windows-specific compiler issues by setting environment variables
    to ensure Inductor can find a working compiler.
    """
    env: dict[str, Any] = {}

    # Memory management for CUDA
    alloc_conf = "garbage_collection_threshold:0.8,max_split_size_mb:128"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", alloc_conf)
    env["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf

    # Config for dynamo
    if hasattr(torch._dynamo, "config"):
        torch._dynamo.config.cache_size_limit = config.cache_size_limit
        torch._dynamo.config.capture_scalar_outputs = True
        if config.verbose:
            torch._dynamo.config.verbose = True

    # Inductor configuration
    try:
        from torch._inductor import config as inductor_config

        # Check for C++ compiler availability on Windows
        import shutil
        has_cl = shutil.which("cl") is not None
        if sys.platform == "win32" and not has_cl:
            # No MSVC compiler - disable C++ codegen and wrapper
            inductor_config.cpp_wrapper = False
            inductor_config.cpp_codegen = False
            os.environ.setdefault("TORCHINDUCTOR_CPP_WRAPPER", "0")
            if GUARD_ENABLED:
                from ._internal.guard import guard, WindWarning
                guard(
                    "WARNING",
                    "C++ compiler (cl.exe) not found on Windows. "
                    "Compilation may fail for CPU subgraphs or graph breaks. "
                    "Consider installing MSVC Build Tools from "
                    "https://visualstudio.microsoft.com/visual-cpp-build-tools/ "
                    "or use mode='default' for safer compilation.",
                    "WiNDCompile",
                    category=WindWarning,
                )

        # Disable max_autotune_gemm on low-SM GPUs to prevent freezes and
        # reduce cache size (max_autotune generates large caches)
        if torch.cuda.is_available():
            try:
                sm_count = torch.cuda.get_device_properties(0).multi_processor_count
                if sm_count <= 32:
                    inductor_config.max_autotune_gemm = False
                    inductor_config.max_autotune = False
            except Exception:
                pass
        else:
            # CPU compilation: disable expensive autotuning
            inductor_config.max_autotune = False
            inductor_config.max_autotune_gemm = False

    except Exception:
        pass

    # Set Triton cache directory to a writable location
    # On Windows, default temp dirs may have issues or run out of space
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not cache_dir:
        cache_dir = os.path.join(
            os.path.expanduser("~"),
            ".wind", "compile_cache"
        )
        try:
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        except Exception:
            pass

    return env


def compile(model: nn.Module, *,
            mode: str = "auto",
            dynamic: bool | None = None,
            fullgraph: bool = False,
            cudagraphs: str = "auto",
            show_progress: bool = True,
            cache_size_limit: int = 64,
            fallback_to_eager: bool = False,
            disable: bool = False,
            warmup: bool = True,
            warmup_input: dict | None = None) -> nn.Module:
    """Smart torch.compile wrapper with automatic mode selection and progress.

    Args:
        model: The nn.Module to compile.
        mode: "auto" selects the best mode based on GPU/model size.
        dynamic: Dynamic shapes (None = auto: False on CUDA for cudagraphs).
        fullgraph: Require single graph (False = allow breaks).
        cudagraphs: "auto" | "on" | "off" — CUDA graph usage.
        show_progress: Show a progress indicator during first compile.
        cache_size_limit: Dynamo cache size limit.
        fallback_to_eager: Return the original module after a compile failure.
            Disabled by default so callers cannot mistake eager fallback for a
            compiled execution path.
        disable: If True, return model unchanged (for debugging).
        warmup: If True, run a dummy forward pass to trigger compilation upfront.
        warmup_input: Optional dict of dummy inputs for warmup. If None,
                      attempts to infer from model (requires forward signature).

    Returns:
        Compiled (or original) model.

    Example::

        model = wind.compile(model, mode="reduce-overhead")
        # or
        model = wind.compile(model)  # auto mode
    """
    global _COMPILE_STATS

    if disable or not hasattr(torch, "compile"):
        return model

    _COMPILE_STATS["total_compiles"] += 1

    config = CompileConfig(
        mode=mode,
        dynamic=dynamic,
        fullgraph=fullgraph,
        cudagraphs=cudagraphs,
        show_progress=show_progress,
        cache_size_limit=cache_size_limit,
        fallback_to_eager=fallback_to_eager,
        disable=disable,
    )

    _setup_compile_env(config)

    # Auto-select mode
    best_mode = _select_best_mode(config, model)
    best_cudagraphs = _select_cudagraphs(config)

    # Auto-select dynamic: static for CUDA graphs
    if config.dynamic is None:
        config.dynamic = False if best_cudagraphs == "on" else True

    # Static analysis before compile
    if GUARD_ENABLED:
        _detect_recompilation_risk(model)

    # Configure cudagraphs
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudagraphs = (best_cudagraphs == "on")

    progress = CompileProgress(stream=sys.stderr)
    if config.show_progress:
        progress.start(type(model).__name__, best_mode)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*recompiled.*")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")

            progress.step("Setting up inductor")

            progress.step("Compiling with torch.inductor")

            compiled_model = torch.compile(
                model,
                mode=best_mode,
                dynamic=config.dynamic,
                fullgraph=config.fullgraph,
                backend=config.backend,
            )

            progress.finish(True)
            _COMPILE_STATS["errors"] = _COMPILE_STATS.get("errors", 0)

            # Post-compile: register recompilation monitoring
            if GUARD_ENABLED and hasattr(compiled_model, "_orig_mod"):
                _check_recompilation_count(compiled_model._orig_mod)

            # Warmup: trigger the first compilation eagerly so errors
            # surface here rather than during training
            if warmup:
                _try_warmup(compiled_model, warmup_input, model, config, progress)

            return compiled_model

    except Exception as e:
        progress.finish(False)
        _COMPILE_STATS["errors"] += 1
        msg = f"wind.compile failed for {type(model).__name__}: {e}"
        guard("ERROR", msg, "WiNDCompile", exc=RuntimeError)

        if config.fallback_to_eager:
            warn_msg = (
                f"Compilation failed, using eager mode. "
                f"Consider: mode='default', fullgraph=False, or check for unsupported ops. "
                f"Raw error: {type(e).__name__}: {str(e)[:200]}"
            )
            guard("WARNING", warn_msg, "WiNDCompile", category=WindWarning)
            return model
        raise RuntimeError(msg) from e


def _try_warmup(compiled_model: nn.Module, warmup_input: dict | None,
                original_model: nn.Module, config: CompileConfig,
                progress: CompileProgress) -> None:
    """Run a dummy forward pass to trigger compilation upfront.

    This surfaces compilation errors here rather than during training.
    """
    progress.step("Warmup: triggering first compilation")

    try:
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r".*recompiled.*")
                warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
                warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")

                if warmup_input is not None:
                    compiled_model(**warmup_input)
                else:
                    # Try to infer a dummy input from model signature
                    # This is best-effort; if we can't, we skip warmup
                    _infer_and_run_warmup(compiled_model, original_model)

    except Exception as e:
        # Warmup failed — re-raise so the outer try/except can catch it
        # and fall back to uncompiled model
        msg = f"Warmup compilation failed for {type(original_model).__name__}"
        if "cl is not found" in str(e) or "Compiler" in str(e):
            long_msg = (
                f"torch.compile requires a C++ compiler (cl.exe) to generate code "
                f"for CPU subgraphs. Install MSVC Build Tools from: "
                f"https://visualstudio.microsoft.com/visual-cpp-build-tools/ "
                f"Select 'Desktop development with C++' workload."
            )
            raise RuntimeError(long_msg) from e
        raise


def _infer_and_run_warmup(compiled_model: nn.Module, original_model: nn.Module) -> None:
    """Best-effort inference of warmup inputs from model structure."""
    import inspect

    sig = inspect.signature(original_model.forward)

    # Try to find input_ids in the signature
    if "input_ids" in sig.parameters:
        # Get vocab size from embedding
        vocab_size = 256
        if hasattr(original_model, "config"):
            vocab_size = original_model.config.vocab_size

        device = next(original_model.parameters()).device
        dtype = next(original_model.parameters()).dtype

        input_ids = torch.randint(0, min(vocab_size, 255), (2, 8), device=device)
        labels = torch.randint(0, min(vocab_size, 255), (2, 8), device=device)

        with torch.inference_mode():
            compiled_model(input_ids=input_ids, labels=labels)
    else:
        raise ValueError(
            "Cannot infer warmup inputs for this model. "
            "Pass warmup_input explicitly to wind.compile()."
        )


def get_compile_stats() -> dict[str, int]:
    """Return compilation statistics."""
    return dict(_COMPILE_STATS)


def reset_compile_stats() -> None:
    """Reset compilation statistics."""
    _COMPILE_STATS.clear()
    _COMPILE_STATS.update({"recompiles": 0, "errors": 0, "total_compiles": 0})
