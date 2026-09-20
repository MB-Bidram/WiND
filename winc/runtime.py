"""WiND Engine runtime: device, dtype, AMP, compilation, and transfer policy.

This module owns all execution policy. It is internal to WiND.
The public API delegates to this module for execution concerns.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

_WIND_PROFILE = os.environ.get("WIND_PROFILE", "") != ""
_PROFILE_ENV = os.environ.get("WIND_DEBUG", "") == "1"


@contextmanager
def _compile_ctx():
    """Suppress non-critical compiler warnings unless profiling/debugging."""
    if _WIND_PROFILE or _PROFILE_ENV:
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*torch\.compile.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*time\.perf_counter.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*recompilation.*",
            category=UserWarning,
        )
        warnings.filterwarnings("ignore", module="torch._dynamo")
        yield


@dataclass(frozen=True)
class HardwareProfile:
    """Configuration for hardware execution.

    Attributes:
        device: Target device (cuda or cpu)
        dtype: Parameter/storage dtype
        compute_dtype: Dtype for autocast compute (defaults to dtype)
        compile: Whether to use torch.compile
        non_blocking: Use non_blocking transfers
        channels_last: Use channels_last memory format
        tf32: Enable TF32 on CUDA
    """

    device: torch.device
    dtype: torch.dtype
    compute_dtype: torch.dtype | None = None
    compile: bool = False
    non_blocking: bool = True
    channels_last: bool = True
    tf32: bool = True

    def __post_init__(self):
        if self.dtype != torch.float32 and self.compute_dtype is None:
            object.__setattr__(self, "compute_dtype", self.dtype)

    @classmethod
    def auto(cls, *, compile: bool = False, fp16: bool = False) -> "HardwareProfile":
        """Auto-detect optimal profile.

        Defaults to bf16 on CUDA (same exponent range as fp32, no GradScaler needed).
        Pass ``fp16=True`` to force FP16 — note this requires GradScaler.
        """
        if torch.cuda.is_available():
            if fp16 and not torch.cuda.is_bf16_supported():
                dtype = torch.float16
            elif torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif fp16:
                dtype = torch.float16
            else:
                dtype = torch.float16
            return cls(
                torch.device("cuda"),
                dtype,
                dtype,
                compile,
                True,
            )
        return cls(
            torch.device("cpu"),
            torch.float32,
            torch.float32,
            False,
            False,
            False,
            False,
        )


def optimize(module: nn.Module, profile: HardwareProfile | None = None) -> nn.Module:
    """Convenience: prepare a module with an OptimizedBackend."""
    return OptimizedBackend(profile).prepare(module)


class OptimizedBackend:
    """Portable execution backend.

    Owns: device placement, dtype policy, AMP/autocast, compilation,
    transfer optimization, synchronization, and timing.

    This is the Engine's central execution policy holder.
    """

    def __init__(self, profile: HardwareProfile | None = None):
        self.profile = profile or HardwareProfile.auto()
        self.is_cuda = self.profile.device.type == "cuda" and torch.cuda.is_available()
        compute_dtype = self.profile.compute_dtype or self.profile.dtype
        self.autocast_enabled = self.is_cuda and compute_dtype in (torch.float16, torch.bfloat16)
        self.scaler_enabled = False  # bf16 needs no GradScaler — no per-step sync
        self._configure_hardware()

    def _configure_hardware(self) -> None:
        """Apply process-safe math/kernel settings once, when available."""
        if self.is_cuda and self.profile.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

    @classmethod
    def auto(cls, *, compile: bool = False, fp16: bool = False) -> "OptimizedBackend":
        return cls(HardwareProfile.auto(compile=compile, fp16=fp16))

    def prepare(self, module: nn.Module) -> nn.Module:
        """Move to device/dtype, apply channels_last, optionally compile."""
        module = module.to(device=self.profile.device)
        if self.profile.dtype != torch.float32:
            module = module.to(dtype=self.profile.dtype)
        if self.is_cuda and self.profile.channels_last:
            try:
                module = module.to(memory_format=torch.channels_last)
            except (RuntimeError, NotImplementedError):
                pass
        if self.profile.compile and hasattr(torch, "compile"):
            with _compile_ctx():
                try:
                    module = torch.compile(module)
                except Exception as exc:
                    raise RuntimeError(
                        "torch.compile failed while preparing the WiNC module; "
                        "the module was not silently downgraded to eager execution"
                    ) from exc
        return module

    def optimizer(self, module: nn.Module | Iterable[torch.Tensor], *, lr: float = 3e-4,
                  weight_decay: float = 0.01, kind: str = "adamw",
                  **kwargs) -> torch.optim.Optimizer:
        """Create a fast optimizer, selecting fused AdamW when supported."""
        source = module.parameters() if isinstance(module, nn.Module) else module
        params = (p for p in source if p.requires_grad)
        if kind.lower() != "adamw":
            return getattr(torch.optim, kind)(params, lr=lr, weight_decay=weight_decay, **kwargs)
        options = dict(lr=lr, weight_decay=weight_decay, **kwargs)
        if self.is_cuda:
            try:
                return torch.optim.AdamW(params, fused=True, **options)
            except (TypeError, RuntimeError):
                pass
        return torch.optim.AdamW(params, **options)

    def checkpoint(self, function, *args, **kwargs):
        """Activation-checkpoint a custom function during training."""
        if not torch.is_grad_enabled():
            return function(*args, **kwargs)
        if kwargs:
            return checkpoint(
                lambda *values: function(*values, **kwargs),
                *args,
                use_reentrant=False,
            )
        return checkpoint(function, *args, use_reentrant=False)

    def clip_grad_norm(self, module: nn.Module, max_norm: float) -> torch.Tensor:
        return torch.nn.utils.clip_grad_norm_(
            (p for p in module.parameters() if p.grad is not None), max_norm
        )

    @contextmanager
    def inference(self):
        """Context for inference mode."""
        with torch.inference_mode():
            yield

    @contextmanager
    def timed(self, name: str, sink: dict[str, float] | None = None):
        """Measure a stage without synchronizing CPU-only execution."""
        if self.is_cuda:
            torch.cuda.synchronize(self.profile.device)
        start = perf_counter()
        yield
        if self.is_cuda:
            torch.cuda.synchronize(self.profile.device)
        if sink is not None:
            sink[name] = sink.get(name, 0.0) + (perf_counter() - start)

    def transfer(self, value: Any) -> Any:
        """Transfer a tensor to the backend device/dtype, skipping no-op transfers."""
        if not isinstance(value, torch.Tensor):
            return value
        # Skip transfer if already on correct device and dtype.
        if (value.device == self.profile.device and
                (not value.is_floating_point() or value.dtype == self.profile.dtype)):
            return value
        dtype = self.profile.dtype if value.is_floating_point() else value.dtype
        return value.to(
            device=self.profile.device,
            dtype=dtype,
            non_blocking=self.profile.non_blocking,
        )

    def autocast(self):
        """Context manager for autocast based on compute_dtype."""
        if self.autocast_enabled:
            dtype = self.profile.compute_dtype or self.profile.dtype
            return torch.autocast(device_type="cuda", dtype=dtype)
        return nullcontext()

    def synchronize(self) -> None:
        """Synchronize device."""
        if self.is_cuda:
            torch.cuda.synchronize(self.profile.device)

    def reset_peak_memory(self) -> None:
        """Reset peak memory statistics (CUDA only)."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()

    def peak_memory_allocated(self) -> int:
        """Return peak allocated memory in bytes."""
        if self.is_cuda:
            return torch.cuda.max_memory_allocated()
        return 0

    def peak_memory_reserved(self) -> int:
        """Return peak reserved memory in bytes."""
        if self.is_cuda:
            return torch.cuda.max_memory_reserved()
        return 0
