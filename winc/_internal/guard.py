"""Wind guard: standardized error/warning reporting and PyTorch-level assertions.

Provides a standard format for all Wind / FlashPKM runtime diagnostics:
    [path:line] Kind:[Subsystem/Component] message

Where Kind is one of ERROR, WARNING, ASSERT, DEPRECATION.

This module is internal to Wind. Public re-exports live in wind/__init__.py.
"""

from __future__ import annotations

import os
import sys
import traceback
import warnings
from typing import Any

import torch
from torch import nn

from torch._dynamo import allow_in_graph

__all__ = [
    "WindError",
    "WindWarning",
    "WindDeprecationWarning",
    "WindAssertError",
    "guard",
    "assert_tensor",
    "assert_finite",
    "assert_dtype",
    "assert_device",
    "assert_shape",
    "assert_shape_compatible",
    "warn_sync",
    "warn_dtype_mismatch",
    "warn_device_mismatch",
    "assert_amp_compatible",
    "assert_no_fp16_master_weights",
    "assert_not_in_hot_path",
    "assert_gradient_has_values",
    "assert_loss_monotonically_decreasing",
    "track_compile_start",
    "check_compile_timeout",
    "track_compile_end",
    "reset_compile_tracker",
    "GUARD_ENABLED",
    "GUARD_STRICT",
    "set_guard_enabled",
    "set_guard_strict",
]

# Runtime value scans (notably ``assert_finite``) require a device-to-host
# decision in eager execution and introduce data-dependent Python control flow
# that Dynamo cannot capture.  They are diagnostics, not model semantics: in
# non-strict mode they only warn.  Keep them explicitly opt-in so production
# execution can use eager and torch.compile hot paths without hidden scans or
# graph breaks.  Set WIND_GUARDS=1 for interactive diagnostics/tests.
GUARD_ENABLED = os.environ.get("WIND_GUARDS", "0") == "1"
GUARD_STRICT = False


def set_guard_enabled(value: bool) -> None:
    global GUARD_ENABLED
    GUARD_ENABLED = value


def set_guard_strict(value: bool) -> None:
    global GUARD_STRICT
    GUARD_STRICT = value


@allow_in_graph
def _find_wind_frame() -> tuple[str, int, str]:
    """Walk the call stack to find the first frame outside this module.

    Uses traceback.extract_stack to remain compatible with torch.compile /
    Dynamo tracing.
    """
    frames = traceback.extract_stack()
    for frame_info in reversed(frames[:-1]):
        filename = frame_info.filename
        if "_internal/guard" not in filename and "guard.py" not in filename:
            return (
                filename,
                frame_info.lineno,
                frame_info.name or "<unknown>",
            )
    if frames:
        last = frames[-2] if len(frames) > 1 else frames[-1]
        return (last.filename, last.lineno, last.name)
    return ("<unknown>", 0, "<unknown>")


def _fmt_location(filename: str, lineno: int) -> str:
    return f"[{filename}:{lineno}]"


def _format_name(name: str) -> str:
    """Format a variable name in PyTorch style."""
    return f"`{name}`"


def guard(kind: str, message: str, subsystem: str = "", *,
          exc: type[Exception] | None = None,
          category: type[Warning] | None = None) -> str:
    """Format and optionally raise/warn a Wind diagnostic.

    Format: ``[path:line] Kind:[Subsystem] message``
    """
    filename, lineno, func = _find_wind_frame()
    tag = f"{kind}:{f'[{subsystem}]' if subsystem else ''}"
    formatted = f"{_fmt_location(filename, lineno)} {tag} {message}"

    if not GUARD_ENABLED:
        return formatted

    if exc is not None and GUARD_STRICT:
        raise exc(formatted)

    if category is not None:
        warnings.warn(formatted, category, stacklevel=3)

    return formatted


class WindError(Exception):
    """Base exception for all Wind/FlashPKM user-facing errors."""


class WindWarning(UserWarning):
    """Base warning category for all Wind/FlashPKM warnings."""


class WindDeprecationWarning(DeprecationWarning):
    """Deprecation notice emitted by Wind."""


class WindAssertError(WindError):
    """Raised when a guarded assertion fires in strict mode."""


def _subsystem_from_obj(obj: Any) -> str:
    if obj is None:
        return "WiNC"
    cls = getattr(obj, "__class__", None) or getattr(obj, "__name__", None)
    if cls:
        return getattr(cls, "__name__", str(cls))
    return "WiNC"


def assert_tensor(value: Any, name: str = "tensor", *,
                  dtype: torch.dtype | None = None,
                  device: torch.device | str | None = None,
                  shape: tuple[int | None, ...] | None = None,
                  min_dims: int | None = None,
                  context: Any = None) -> None:
    """Assert *value* is a tensor, optionally checking dtype/device/shape.

    In strict mode this raises :class:`WindAssertError`.  Otherwise it emits
    a :class:`RuntimeWarning` and returns.

    Produces PyTorch-style messages like::

        AssertionError: expected Tensor as element 0 in argument #0, but got list
    """
    n = _format_name(name)
    if not isinstance(value, torch.Tensor):
        actual_type = type(value).__name__
        msg = f"expected Tensor as element 0 in argument #{0}, but got {actual_type}"
        guard("ASSERT", msg, _subsystem_from_obj(context), exc=WindAssertError)
        return

    sub = _subsystem_from_obj(context)
    if dtype is not None and value.dtype != dtype:
        msg = (
            f"{n} must have dtype {dtype}, but has dtype {value.dtype}"
        )
        guard("ASSERT", msg, sub, exc=WindAssertError)

    if device is not None:
        expected_dev = torch.device(device) if isinstance(device, str) else device
        if value.device != expected_dev:
            msg = f"{n} must be on device '{expected_dev}', but is on device '{value.device}'"
            guard("ASSERT", msg, sub, exc=WindAssertError)

    if shape is not None:
        actual = tuple(value.shape)
        mismatches = [
            (i, s, a) for i, (s, a) in enumerate(zip(shape, actual))
            if s is not None and s != a
        ]
        if mismatches:
            expected_str = tuple(s if s is not None else "?" for s in shape)
            msg = (
                f"{n} must have shape {expected_str}, but has shape {actual}"
            )
            guard("ASSERT", msg, sub, exc=WindAssertError)

    if min_dims is not None and value.ndim < min_dims:
        msg = f"{n} must have at least {min_dims} dimensions, but has {value.ndim}"
        guard("ASSERT", msg, sub, exc=WindAssertError)


def assert_shape(x: torch.Tensor, expected: tuple[int | None, ...], name: str = "tensor",
                 context: Any = None) -> None:
    """Assert tensor `x` has shape matching `expected` (None = any)."""
    n = _format_name(name)
    if not isinstance(x, torch.Tensor):
        guard("ASSERT", f"expected Tensor for {n}, but got {type(x).__name__}",
              _subsystem_from_obj(context), exc=WindAssertError)
        return
    actual = tuple(x.shape)
    if len(actual) != len(expected):
        expected_str = tuple(s if s is not None else "?" for s in expected)
        guard("ASSERT", f"{n} must have shape {expected_str}, but has shape {actual}",
              _subsystem_from_obj(context), exc=WindAssertError)
        return
    mismatches = [i for i, (e, a) in enumerate(zip(expected, actual)) if e is not None and e != a]
    if mismatches:
        expected_str = tuple(s if s is not None else "?" for s in expected)
        guard("ASSERT", f"{n} must have shape {expected_str}, but has shape {actual}",
              _subsystem_from_obj(context), exc=WindAssertError)


def assert_shape_compatible(a: torch.Tensor, b: torch.Tensor, name_a: str = "a", name_b: str = "b",
                            context: Any = None) -> None:
    """Assert tensors are broadcast-compatible for elementwise ops."""
    n_a = _format_name(name_a)
    n_b = _format_name(name_b)
    try:
        torch.broadcast_shapes(a.shape, b.shape)
    except RuntimeError as e:
        msg = (
            f"the shapes for {n_a} {tuple(a.shape)} and {n_b} {tuple(b.shape)} "
            f"are not broadcastable: {e}"
        )
        guard("ASSERT", msg, _subsystem_from_obj(context), exc=WindAssertError)


def assert_finite(t: torch.Tensor, name: str = "tensor", context: Any = None) -> None:
    """Assert all elements of `t` are finite (no NaN/Inf)."""
    if not isinstance(t, torch.Tensor):
        guard("ASSERT", f"expected Tensor for {_format_name(name)}, but got {type(t).__name__}",
              _subsystem_from_obj(context), exc=WindAssertError)
        return
    if not torch.isfinite(t).all():
        n_bad = int(((~torch.isfinite(t)).sum()).item())
        total = t.numel()
        n = _format_name(name)
        msg = (
            f"{n} contains {n_bad} non-finite values out of {total} "
            f"(NaN or Inf detected)"
        )
        guard("ASSERT", msg, _subsystem_from_obj(context), exc=WindAssertError)


def assert_dtype(t: torch.Tensor, expected: torch.dtype, name: str = "tensor", context: Any = None) -> None:
    if not isinstance(t, torch.Tensor):
        guard("ASSERT", f"expected Tensor for {_format_name(name)}, but got {type(t).__name__}",
              _subsystem_from_obj(context), exc=WindAssertError)
        return
    if t.dtype != expected:
        n = _format_name(name)
        guard("ASSERT", f"{n} must have dtype {expected}, but has dtype {t.dtype}",
              _subsystem_from_obj(context), exc=WindAssertError)


def assert_device(t: torch.Tensor, expected: torch.device | str, name: str = "tensor", context: Any = None) -> None:
    if not isinstance(t, torch.Tensor):
        guard("ASSERT", f"expected Tensor for {_format_name(name)}, but got {type(t).__name__}",
              _subsystem_from_obj(context), exc=WindAssertError)
        return
    exp = torch.device(expected) if isinstance(expected, str) else expected
    if t.device != exp:
        n = _format_name(name)
        guard("ASSERT", f"{n} must be on device '{exp}', but is on device '{t.device}'",
              _subsystem_from_obj(context), exc=WindAssertError)


def warn_sync(operation: str = "device sync", context: Any = None) -> None:
    """Emit a warning about a potential sync point in the hot path."""
    msg = (
        f"Potential {operation} detected in forward/backward path. "
        "If this runs inside a compiled or hot loop, it may serialize kernel launches "
        "and reduce throughput. Consider deferring to an epoch boundary or using "
        "a non-blocking check."
    )
    guard("WARNING", msg, _subsystem_from_obj(context), category=WindWarning)


def warn_dtype_mismatch(actual: torch.dtype, expected: torch.dtype, name: str = "tensor",
                        context: Any = None) -> None:
    n = _format_name(name)
    msg = f"{n} has dtype {actual}, which does not match the expected dtype {expected}"
    if actual.is_floating_point and expected.is_floating_point:
        msg += " — mixed-precision mismatch may cause silent accuracy loss or kernel fallbacks"
    guard("WARNING", msg, _subsystem_from_obj(context), category=WindWarning)


def warn_device_mismatch(actual: torch.device, expected: torch.device, name: str = "tensor",
                         context: Any = None) -> None:
    n = _format_name(name)
    msg = f"{n} is on device '{actual}', which does not match the expected device '{expected}'"
    guard("WARNING", msg, _subsystem_from_obj(context), category=WindWarning)


_warn_cache: dict[str, int] = {}


def _dedup_warn(message: str, category: type[Warning] = WindWarning, max_times: int = 1) -> bool:
    """Emit a warning at most `max_times` times per process to avoid log spam."""
    key = message[:200]
    count = _warn_cache.get(key, 0)
    if count >= max_times:
        return False
    _warn_cache[key] = count + 1
    warnings.warn(message, category, stacklevel=3)
    return True


# ── Compile-specific guards ──────────────────────────────────────────

compile_loop_tracker: dict[int, dict[str, Any]] = {}
_COMPILE_MAX_RECOMPILATIONS = 100
_COMPILE_MAX_TIME_PER_COMPILE = 300.0  # 5 minutes


def track_compile_start(key: str, *, max_recompiles: int = _COMPILE_MAX_RECOMPILATIONS) -> None:
    """Track a compilation attempt for infinite loop detection.

    Always raises WindError if the recompilation limit is exceeded, regardless
    of GUARD_STRICT setting, since infinite recompilation loops crash the process.
    """
    import time
    tracker = compile_loop_tracker.setdefault(id(key), {})
    tracker["start_time"] = time.perf_counter()
    tracker["compiles"] = tracker.get("compiles", 0) + 1
    tracker["max_recompiles"] = max_recompiles

    if tracker["compiles"] > max_recompiles:
        msg = (
            f"torch.compile has exceeded {max_recompiles} recompilations "
            f"(attempt #{tracker['compiles']}). "
            "This may indicate an infinite recompilation loop caused by: "
            "1) dynamic shapes in the hot path, 2) .item()/.tolist() calls, "
            "3) conditional branches on runtime values, 4) tensor.data mutations. "
            "Consider using static=True or disabling compilation."
        )
        raise WindError(f"[{tracker.get('filename', '?')}:{tracker.get('lineno', '?')}] "
                        f"ASSERT:[WiNDCompile] {msg}")


def check_compile_timeout(key: str) -> None:
    """Check if a compilation has been running too long."""
    import time
    tracker = compile_loop_tracker.get(id(key))
    if tracker is None:
        return
    elapsed = time.perf_counter() - tracker.get("start_time", 0)
    if elapsed > _COMPILE_MAX_TIME_PER_COMPILE:
        msg = (
            f"torch.compile has been running for {elapsed:.1f}s, "
            f"exceeding the {int(_COMPILE_MAX_TIME_PER_COMPILE)}s safety limit. "
            "This may indicate an infinite compilation loop. "
            "Consider reducing cache_size_limit or checking for unsupported ops."
        )
        raise WindError(f"[WiNDCompile] {msg}")


def track_compile_end(key: str) -> None:
    """Mark compilation as complete for a tracked key."""
    import time
    tracker = compile_loop_tracker.pop(id(key), None)
    if tracker is not None:
        tracker["end_time"] = time.perf_counter()


def reset_compile_tracker() -> None:
    """Reset all compilation trackers."""
    compile_loop_tracker.clear()


# ── Mixed precision / dtype guards ───────────────────────────────────

def assert_amp_compatible(dtype: torch.dtype, context: Any = None) -> None:
    """Assert that a dtype is suitable for AMP (fp16 or bf16)."""
    if dtype not in (torch.float16, torch.bfloat16):
        msg = (
            f"AMP is enabled but model dtype is {dtype}. "
            "AMP requires float16 or bfloat16 for compute. "
            "Use bf16 for best stability (same exponent range as fp32)."
        )
        guard("ASSERT", msg, _subsystem_from_obj(context), exc=WindAssertError)


def assert_no_fp16_master_weights(context: Any = None) -> None:
    """Warn if using fp16 with master weights (requires GradScaler)."""
    msg = (
        "Using float16 with master_weights=True is an unusual configuration. "
        "BF16 is preferred for stability (same exponent range as fp32). "
        "If you encounter overflow, consider switching to bf16."
    )
    guard("WARNING", msg, _subsystem_from_obj(context), category=WindWarning)


# ── Training loop guards ─────────────────────────────────────────────

def assert_not_in_hot_path(operation: str, context: Any = None) -> None:
    """Assert that an operation is not being called in a hot path."""
    msg = (
        f"Potential sync operation detected: {operation}. "
        "Calling .item(), .cpu(), or bool() on device tensors in a training loop "
        "creates an implicit cudaStreamSynchronize which stalls the GPU pipeline. "
        "Use tensor-based operations or defer to an epoch boundary."
    )
    guard("WARNING", msg, _subsystem_from_obj(context), category=WindWarning)


def assert_gradient_has_values(module: nn.Module, context: Any = None) -> None:
    """Assert that at least one parameter has a gradient."""
    has_grad = any(p.grad is not None for p in module.parameters() if p.requires_grad)
    if not has_grad:
        msg = (
            "No gradients found on any trainable parameters. "
            "This may indicate: 1) missing loss.backward(), 2) detached graph, "
            "3) no trainable params, 4) checkpoint loading issue."
        )
        guard("ASSERT", msg, _subsystem_from_obj(context), exc=WindAssertError)


def assert_loss_monotonically_decreasing(loss: float, prev_loss: float | None,
                                          step: int, threshold: float = 0.01) -> bool:
    """Assert loss is decreasing (warn, not error, on divergence)."""
    if prev_loss is None or step == 0:
        return True
    if loss > prev_loss + threshold:
        msg = (
            f"Loss increased from {prev_loss:.4f} to {loss:.4f} at step {step}. "
            "This may indicate: 1) learning rate too high, 2) data corruption, "
            "3) numerical instability (overflow/NaN)."
        )
        guard("WARNING", msg, "LMTrainer", category=WindWarning)
        return False
    return True
