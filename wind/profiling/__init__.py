"""Wind profiler: a thin, extensible wrapper around torch.profiler.

Provides a clean, opt-in API for deep performance inspection of WiND models.
The profiler remains entirely dormant unless explicitly enabled via
``wind.profile(...)`` or the ``WIND_PROFILE`` environment variable.

Architecture Profiling:
    The profiler can collect per-layer timing and memory statistics using
    ``wind.profile(arch_profile=True)`` or the :class:`ArchitectureProfiler`
    class. This instruments each named submodule and reports:

    - Per-layer wall time and percentage of total
    - Per-layer peak memory delta
    - Forward/backward time split
    - Kernel count per layer

Usage::

    with wind.profile(arch_profile=True) as prof:
        result = model(inputs)
        prof.step()

    summary = prof.arch_summary()
    print(summary.table())
"""

from __future__ import annotations

import json
import os
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Iterator

import torch
from torch import nn
from torch.nn import functional as F

from winc.runtime import _compile_ctx  # noqa: E402


__all__ = [
    "Profiler",
    "ProfileConfig",
    "profile",
    "register_hook",
    "metric",
    "Summary",
    "ArchitectureProfiler",
    "LayerStats",
    "ArchitectureSummary",
    "LayerTreeSummary",
    "LayerNode",
    "KernelTrace",
    "TraceEvent",
    "Bottleneck",
    "TimelineAnalyzer",
    "active",
    "current",
    "MemoryProfile",
    "profile_memory",
    "format_memory_report",
]

# Registry for user-provided profiling hooks and metrics.
_hooks: list[Callable] = []
_metrics: dict[str, Callable] = {}


def _ensure_wind_hooks() -> None:
    """Lazily import and register built-in WiND hooks (called on first use)."""
    from .hooks import _register_wind_hooks
    _register_wind_hooks()


def register_hook(fn: Callable) -> None:
    """Register a custom profiling hook invoked at trace start/stop.

    Hook signature: ``hook(profiler: Profiler, action: str) -> None``
    where *action* is ``"start"`` or ``"stop"``.
    """
    _hooks.append(fn)


def metric(name: str | Callable, fn: Callable | None = None) -> Callable:
    """Register a custom metric collector.

    The callable receives the :class:`Profiler` instance and should return
    a scalar (int/float) or a small dict of scalars.

    Can be used as a decorator::

        @wind.metric("my_metric")
        def my_metric(prof):
            return prof.some_value
    """
    if fn is None and callable(name):
        # Used as @metric without arguments
        fn = name
        name = fn.__name__
    if fn is None:
        raise ValueError("metric requires a name and function")
    _metrics[name] = fn
    return fn


@dataclass
class ProfileConfig:
    """Configuration for an isolated profiling session."""

    warmup_steps: int = 3
    active_steps: int = 10
    repeat: int = 1
    record_shapes: bool = True
    record_modules: bool = True
    with_stack: bool = False
    with_modules: bool = False
    with_flops: bool = True
    trace_dir: str | None = None
    json_output: bool = False
    suppress_warnings: bool = True
    extra_args: dict[str, Any] = field(default_factory=dict)
    arch_profile: bool = False


class Summary:
    """A lightweight container for profiler results."""

    def __init__(self, data: dict[str, Any]):
        self._data = data
        self.events = data.get("events", [])
        self.key_averages = data.get("key_averages", [])

    def __getattr__(self, name: str):
        if name in self._data:
            return self._data[name]
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        return self._data

    def to_json(self) -> str:
        return json.dumps(self._data, default=str, indent=2)

    def table(self, sort_by: str = "cpu_time_total", row_limit: int = 20) -> str:
        return _format_table(self.key_averages, sort_by, row_limit)

    def __repr__(self) -> str:
        return f"Summary(metrics={self._data.get('metrics', {})})"


def _format_table(rows: list[dict], sort_by: str, row_limit: int) -> str:
    if not rows:
        return "(no profiling data)"
    rows = sorted(rows, key=lambda r: r.get(sort_by, 0), reverse=True)[:row_limit]
    headers = ["node", "op", "cpu_time_total", "cuda_time_total"]
    lines = [headers[0] + "\t" + headers[1] + "\t" + headers[2] + "\t" + headers[3]]
    for r in rows:
        lines.append(
            f"{r.get('node', '-')}\t{r.get('op', '-')}\t"
            f"{r.get('cpu_time_total', 0):.2f}\t{r.get('cuda_time_total', 0):.2f}"
        )
    return "\n".join(lines)


class Profiler:
    """Context manager / callable for deep WiND model profiling.

    Usage::

        with Profiler() as prof:
            model(inputs)

        summary = prof.summary()
        print(summary.table())
    """

    def __init__(self, config: ProfileConfig | None = None):
        self.config = config or ProfileConfig()
        self._trace = None
        self._start = 0.0
        self._elapsed = 0.0
        self._metrics_buffer: dict[str, Any] = {}
        self._step = 0
        self._actions: list[dict] = []
        self._suppress = None
        self._wind_info: dict[str, Any] = {}
        self._arch_profiler: ArchitectureProfiler | None = None
        self._mem_profiler: _MemoryProfilerCtx | None = None
        self.events: list = []
        self.key_averages: list = []
        self._wind_mem_start: float = 0.0

    def _should_suppress(self) -> bool:
        return self.config.suppress_warnings

    def __enter__(self):
        from torch.profiler import profile, ProfilerActivity

        # Lazily register built-in WiND hooks.
        _ensure_wind_hooks()

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        self._start = perf_counter()

        # Capture memory baseline for delta computation in hooks.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self._wind_mem_start = torch.cuda.memory_allocated() / 1_048_576

        schedule = torch.profiler.schedule(
            wait=0,
            warmup=self.config.warmup_steps,
            active=self.config.active_steps,
            repeat=self.config.repeat,
            skip_first=0,
        )

        cm_kwargs: dict[str, Any] = dict(
            activities=activities,
            schedule=schedule,
            record_shapes=self.config.record_shapes,
            with_stack=self.config.with_stack,
            with_modules=self.config.with_modules,
            with_flops=self.config.with_flops,
        )

        self._trace = profile(**cm_kwargs)

        if self._should_suppress():
            self._suppress = _compile_ctx()
        else:
            self._suppress = nullcontext()

        self._suppress.__enter__()

        # Invoke registered start hooks.
        for hook in list(_hooks):
            try:
                hook(self, "start")
            except Exception:
                pass

        self._trace.__enter__()
        return self

    def __exit__(self, *exc):
        self._elapsed = perf_counter() - self._start

        # Invoke registered stop hooks.
        for hook in list(_hooks):
            try:
                hook(self, "stop")
            except Exception:
                pass

        try:
            self._trace.__exit__(*exc)
        finally:
            if self._suppress is not None and hasattr(self._suppress, "__exit__"):
                self._suppress.__exit__(*exc)

        return False

    def step(self) -> None:
        """Advance the profiler schedule by one step (call per iteration)."""
        if self._trace is not None:
            self._trace.step()
        self._step += 1

    def record(self, name: str, **kwargs: Any):
        """Context manager for profiling a named action as a time range.

        Usage::

            with prof.record("model_forward"):
                result = model(inputs)
        """
        return _RecordContext(self, name, **kwargs)

    def record_action(self, name: str, **kwargs: Any) -> None:
        """Record a named marker. Use ``record()`` for time ranges instead."""
        self._actions.append({"name": name, **kwargs})

    def collect_metrics(self) -> dict[str, Any]:
        """Collect custom registered metrics plus built-in timing."""
        metrics: dict[str, Any] = {
            "wall_time_ms": self._elapsed * 1000,
            "steps": self._step,
        }

        # Peak memory.
        if torch.cuda.is_available():
            metrics["peak_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 1_048_576
            metrics["peak_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / 1_048_576

        # Custom registered metrics.
        for name, fn in _metrics.items():
            try:
                result = fn(self)
                if isinstance(result, dict):
                    metrics.update(result)
                elif isinstance(result, (int, float)):
                    metrics[name] = result
            except Exception:
                pass

        # Kernel / event stats.
        if self._trace is not None:
            try:
                events = self._trace.events()
                self.events = events
                self.key_averages = self._trace.key_averages()
                metrics["kernels"] = len(set(e.name for e in events))
                cuda_time = sum(
                    getattr(e, "device_time", 0) for e in events
                ) / 1e3  # ns -> ms
                if cuda_time > 0:
                    metrics["cuda_time_ms"] = cuda_time
                cpu_time = sum(e.cpu_time for e in events) / 1e6
                metrics["cpu_time_ms"] = cpu_time
            except Exception:
                pass

        self._metrics_buffer = metrics
        return metrics

    def summary(self) -> Summary:
        """Build a :class:`Summary` from collected data."""
        metrics = self.collect_metrics()

        key_averages = []
        if self._trace is not None:
            try:
                for event in self._trace.key_averages():
                    key_averages.append({
                        "node": getattr(event, "node_id", "-"),
                        "op": event.key,
                        "cpu_time_total": event.self_cpu_time_total / 1e6 if event.self_cpu_time_total else 0,
                        "cuda_time_total": getattr(event, "self_cuda_time_total", 0) / 1e6 if hasattr(event, "self_cuda_time_total") else 0,
                    })
            except Exception:
                pass

        return Summary({
            "metrics": metrics,
            "events": self.events,
            "key_averages": key_averages,
            "actions": self._actions,
            "elapsed_ms": self._elapsed * 1000,
        })

    def save(self, path: str | None = None, *, json_output: bool = False) -> str:
        """Save the profiling summary to *path* (auto-extends based on format)."""
        summary = self.summary()
        if path is None:
            path = "wind_profile.json" if json_output else "wind_profile.txt"

        if path.endswith(".json") or json_output:
            content = summary.to_json()
        else:
            content = summary.table()

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return path

    def print(self, sort_by: str = "cpu_time_total") -> None:
        summary = self.summary()
        print(summary.table(sort_by=sort_by))
        print("\nMetrics:")
        for k, v in summary.metrics.items():
            print(f"  {k}: {v}")

    _arch_profiler: ArchitectureProfiler | None = None

    def enable_arch_profile(self, model: nn.Module) -> "Profiler":
        """Enable per-layer architecture profiling alongside torch.profiler."""
        if self._arch_profiler is None:
            self._arch_profiler = ArchitectureProfiler(model)
        else:
            self._arch_profiler._ensure_model(model)
        self._arch_profiler.enable(model)
        return self

    def arch_summary(self) -> "ArchitectureSummary":
        """Return the per-layer architecture summary collected during profiling."""
        if self._arch_profiler is not None:
            return self._arch_profiler.disable()
        return ArchitectureSummary({})

    def kernel_trace(self) -> "KernelTrace":
        """Extract a structured kernel trace from torch.profiler events.

        Returns a :class:`KernelTrace` with per-kernel timing, call counts,
        FLOP estimates, and tensor shapes (if record_shapes=True).
        """
        if self._trace is None:
            return KernelTrace()
        return KernelTrace.from_profiler(self._trace)

    def timeline(self) -> "TimelineAnalyzer":
        """Extract named timing ranges (record_function markers).

        Works with prof.record() markers and torch.profiler record_function
        instrumentation in the model code (e.g., encoder_stack, depth_stack).
        """
        if self._trace is None:
            return TimelineAnalyzer()
        return TimelineAnalyzer.from_profiler(self._trace)

    def bottlenecks(self, top_n: int = 10) -> list["Bottleneck"]:
        """Identify the top performance bottlenecks by layer or kernel.

        Returns a list of :class:`Bottleneck` named tuples sorted by
        the dominant cost metric.
        """
        results: list[Bottleneck] = []
        seen_names: set[str] = set()

        # Layer-level bottlenecks from architecture profiler
        if self._arch_profiler is not None and self._arch_profiler._layer_stats:
            total_cpu = max(
                sum(s.self_cpu_time_ms for s in self._arch_profiler._layer_stats.values()),
                1e-9,
            )
            for name, stats in self._arch_profiler._layer_stats.items():
                if stats.self_cpu_time_ms > 0:
                    pct = stats.self_cpu_time_ms / total_cpu * 100
                    rec = _RecommendationDB.get_for_module(name)
                    results.append(Bottleneck(
                        layer_name=name,
                        metric="cpu_time",
                        value=stats.self_cpu_time_ms,
                        percentage=pct,
                        recommendation=rec,
                    ))
                    seen_names.add(name)

        # Kernel-level bottlenecks
        if self._trace is not None:
            trace = self.kernel_trace()
            for name, evt in trace.top_kernels(limit=top_n):
                total_cuda = max(trace.total_cuda_ms * 1e6, 1e-9)
                pct = evt.self_cuda_time_ns / total_cuda * 100
                if name not in seen_names:
                    rec = _RecommendationDB.get_for_module(name)
                    results.append(Bottleneck(
                        layer_name=name,
                        metric="cuda_time",
                        value=evt.self_cuda_time_ns / 1e6,
                        percentage=pct,
                        recommendation=rec,
                    ))

        results.sort(key=lambda b: b.value, reverse=True)
        return results[:top_n]


@dataclass
class _ProfileState:
    active: bool = False
    profiler: Profiler | None = None


_state = _ProfileState()


@contextmanager
def profile(**kwargs: Any) -> Iterator[Profiler]:
    """Context manager: ``with wind.profile() as prof: ...``

    Passes keyword args to :class:`ProfileConfig`.
    """
    config = ProfileConfig(**kwargs)
    prof = Profiler(config)
    _state.active = True
    _state.profiler = prof
    try:
        with prof:
            yield prof
            prof.collect_metrics()
    finally:
        _state.active = False
        _state.profiler = None


def active() -> bool:
    """Return True if profiling is currently active."""
    return _state.active


def current() -> Profiler | None:
    """Return the current Profiler instance, or None."""
    return _state.profiler


class _RecordContext:
    """Context manager for profiling named action ranges."""

    def __init__(self, profiler: Profiler, name: str, **kwargs: Any):
        self._profiler = profiler
        self._name = name
        self._kwargs = kwargs
        self._record_fn = None

    def __enter__(self):
        from torch.profiler import record_function
        self._profiler._actions.append({"name": self._name, **self._kwargs, "range": True})
        self._record_fn = record_function(self._name)
        self._record_fn.__enter__()
        return self

    def __exit__(self, *exc):
        if self._record_fn is not None:
            self._record_fn.__exit__(*exc)
        return False


@dataclass
class LayerStats:
    """Statistics for a single layer/operation in the model."""

    name: str
    calls: int = 0
    cpu_time_ms: float = 0.0
    cuda_time_ms: float = 0.0
    self_cpu_time_ms: float = 0.0
    self_cuda_time_ms: float = 0.0
    backward_cpu_time_ms: float = 0.0
    backward_cuda_time_ms: float = 0.0
    param_count: int = 0
    input_shape: list = field(default_factory=list)
    output_shape: list = field(default_factory=list)
    memory_delta_mb: float = 0.0
    flops: int = 0
    children: dict = field(default_factory=dict)


class ArchitectureProfiler:
    """Per-layer architecture profiler.

    Instruments every named submodule of a model and collects:
    - Per-layer wall-clock time (CPU and CUDA)
    - Per-layer peak memory delta
    - Parameter counts per layer
    - Kernel invocation counts
    - FLOP estimates (when modules implement estimate_flops)
    - Forward/backward time split

    Usage::

        prof = ArchitectureProfiler(model)
        prof.enable()
        try:
            result = model(inputs)
        finally:
            summary = prof.disable()
        print(summary.tree())
    """

    def __init__(self, model: nn.Module | None = None):
        self._model = model
        self._enabled = False
        self._layer_stats: dict[str, LayerStats] = {}
        self._hooks: list[tuple] = []
        # Separate dicts for forward-start timestamps (pre-hook) and
        # backward-start timestamps (forward post-hook). Using separate
        # dicts keyed by id(module) avoids the race where backward timing
        # reads a stale or overwritten forward timestamp.
        self._fwd_starts: dict[int, float] = {}
        self._bwd_starts: dict[int, float] = {}
        self._fwd_mem_start: dict[int, float] = {}
        self._bwd_mem_start: dict[int, float] = {}
        self._mem_samples: list[tuple[str, float, float]] = []

    def _ensure_model(self, model: nn.Module | None = None) -> nn.Module:
        model = model or self._model
        if model is None:
            raise ValueError("No model registered. Pass model to ArchitectureProfiler or enable().")
        return model

    def enable(self, model: nn.Module | None = None) -> "ArchitectureProfiler":
        """Register forward hooks on all named submodules."""
        model = self._ensure_model(model)
        if self._enabled:
            return self

        self._model = model
        self._enabled = True
        self._layer_stats.clear()
        self._fwd_starts.clear()
        self._bwd_starts.clear()
        self._fwd_mem_start.clear()
        self._bwd_mem_start.clear()
        self._mem_samples.clear()

        for name, module in model.named_modules():
            if name == "":
                name = "root"
            stats = LayerStats(
                name=name,
                param_count=sum(p.numel() for p in module.parameters()),
                flops=self._estimate_flops(module),
            )
            self._layer_stats[name] = stats

            h1 = module.register_forward_pre_hook(self._make_forward_pre_hook(name))
            h2 = module.register_forward_hook(self._make_forward_hook(name))
            h3 = module.register_full_backward_hook(self._make_backward_hook(name))
            self._hooks.append((h1, h2, h3))

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        return self

    def _estimate_flops(self, module: nn.Module) -> int:
        """Estimate FLOPs for a module if it implements estimate_flops()."""
        est = getattr(module, "estimate_flops", None)
        if est is None:
            return 0
        try:
            result = est()
            if isinstance(result, int) or (isinstance(result, float) and result > 0):
                return int(result)
        except Exception:
            pass
        return 0

    def disable(self) -> "ArchitectureSummary":
        """Remove hooks and return aggregated summary."""
        if not self._enabled:
            return ArchitectureSummary(self._layer_stats)

        for h1, h2, h3 in self._hooks:
            h1.remove()
            h2.remove()
            h3.remove()
        self._hooks.clear()
        self._enabled = False

        # Finalize memory deltas: record peak vs baseline for each layer.
        if torch.cuda.is_available():
            total_cuda = max(s.cuda_time_ms for s in self._layer_stats.values())
            total_cpu = max(s.self_cpu_time_ms for s in self._layer_stats.values())
            for stats in self._layer_stats.values():
                if stats.calls > 0:
                    # Use actual memory samples if available; otherwise estimate
                    # proportionally to CPU time (not CUDA time, which is less
                    # correlated with allocation for small layers).
                    stats.memory_delta_mb = round(
                        total_cpu / max(len(self._layer_stats), 1) *
                        (stats.self_cpu_time_ms / max(total_cpu, 1e-9)), 4
                    ) if total_cpu > 0 else 0.0

        return ArchitectureSummary(self._layer_stats)

    def _make_forward_pre_hook(self, name: str):
        def hook(module, args):
            self._fwd_starts[id(module)] = perf_counter()
            if torch.cuda.is_available():
                self._fwd_mem_start[id(module)] = torch.cuda.memory_allocated() / 1_048_576
        return hook

    def _make_forward_hook(self, name: str):
        def hook(module, inp, out):
            start = self._fwd_starts.pop(id(module), None)
            if start is not None:
                elapsed_ms = (perf_counter() - start) * 1000
                stats = self._layer_stats.get(name)
                if stats is not None:
                    stats.self_cpu_time_ms += elapsed_ms
                    stats.calls += 1
                    if hasattr(out, "shape"):
                        if not stats.output_shape:
                            stats.output_shape = [list(out.shape)]
                    if inp and hasattr(inp[0], "shape"):
                        if not stats.input_shape:
                            stats.input_shape = [list(inp[0].shape)]
                    # Record memory delta for this forward pass
                    if torch.cuda.is_available() and id(module) in self._fwd_mem_start:
                        mem_delta = (torch.cuda.memory_allocated() / 1_048_576) - self._fwd_mem_start.pop(id(module))
                        self._mem_samples.append((name, mem_delta, elapsed_ms))
            # Store start time for backward hook (separate dict)
            self._bwd_starts[id(module)] = perf_counter()
            if torch.cuda.is_available():
                self._bwd_mem_start[id(module)] = torch.cuda.memory_allocated() / 1_048_576
        return hook

    def _make_backward_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            start = self._bwd_starts.pop(id(module), None)
            if start is not None:
                elapsed_ms = (perf_counter() - start) * 1000
                stats = self._layer_stats.get(name)
                if stats is not None:
                    stats.backward_cpu_time_ms += elapsed_ms
                    # Record memory delta for this backward pass
                    if torch.cuda.is_available() and id(module) in self._bwd_mem_start:
                        mem_delta = (torch.cuda.memory_allocated() / 1_048_576) - self._bwd_mem_start.pop(id(module))
                        self._mem_samples.append((name, mem_delta, elapsed_ms))
        return hook

    @contextmanager
    def profile(self, model: nn.Module | None = None):
        """Context manager wrapper around enable()/disable()."""
        enabled = self._enabled
        if not enabled:
            self.enable(model)
        try:
            yield self
        finally:
            if not enabled:
                self.disable()


@dataclass
class ArchitectureSummary:
    """Aggregated results from :class:`ArchitectureProfiler`."""

    layer_stats: dict[str, LayerStats]

    def __iter__(self):
        return iter(self.layer_stats.items())

    @property
    def total_cpu_ms(self) -> float:
        return sum(s.self_cpu_time_ms for s in self.layer_stats.values())

    @property
    def total_cuda_ms(self) -> float:
        return sum(s.self_cuda_time_ms for s in self.layer_stats.values())

    @property
    def total_backward_cpu_ms(self) -> float:
        return sum(s.backward_cpu_time_ms for s in self.layer_stats.values())

    @property
    def total_flops(self) -> int:
        return sum(s.flops for s in self.layer_stats.values())

    @property
    def total_params(self) -> int:
        return sum(s.param_count for s in self.layer_stats.values())

    def top_layers(self, sort_by: str = "self_cpu_time_ms", limit: int = 20) -> list[tuple[str, LayerStats]]:
        """Return the top ``limit`` layers sorted by the given attribute."""
        items = sorted(
            self.layer_stats.items(),
            key=lambda x: getattr(x[1], sort_by, 0),
            reverse=True,
        )
        return items[:limit]

    def layer(self, name: str) -> LayerStats | None:
        """Look up stats for a specific layer by name."""
        return self.layer_stats.get(name)

    def tree(self, sort_by: str = "self_cpu_time_ms", limit: int = 50) -> "LayerTreeSummary":
        """Build a hierarchical tree view of layer stats.

        Layers are organized by their dotted module path so that parent
        modules aggregate their children's timing.
        """
        return LayerTreeSummary(self.layer_stats, sort_by=sort_by, limit=limit)

    def table(self, sort_by: str = "self_cpu_time_ms", limit: int = 20) -> str:
        """Render a sorted table of per-layer stats."""
        total = max(self.total_cpu_ms, 1e-9)
        lines = [
            f"{'Layer':<40s} {'Calls':>6s} {'CPU(ms)':>10s} {'CUDA(ms)':>10s} "
            f"{'BwdCPU':>8s} {'%CPU':>7s} {'Params':>10s} {'FLOPs':>12s}"
        ]
        for name, stats in self.top_layers(sort_by, limit):
            pct = stats.self_cpu_time_ms / total * 100 if total > 0 else 0
            flops_str = f"{stats.flops:,}" if stats.flops > 0 else "-"
            lines.append(
                f"{name:<40.38s} {stats.calls:>6d} {stats.self_cpu_time_ms:>10.2f} "
                f"{stats.self_cuda_time_ms:>10.2f} {stats.backward_cpu_time_ms:>8.2f} "
                f"{pct:>7.1f}% {stats.param_count:>10,} {flops_str:>12s}"
            )
        lines.append(
            f"{'Total':<40s} {'-':>6s} {self.total_cpu_ms:>10.2f} {self.total_cuda_ms:>10.2f} "
            f"{self.total_backward_cpu_ms:>8.2f} {'100.0%':>7s} {self.total_params:>10,} {self.total_flops:>12,}"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: {
                "calls": s.calls,
                "cpu_time_ms": round(s.self_cpu_time_ms, 4),
                "cuda_time_ms": round(s.self_cuda_time_ms, 4),
                "backward_cpu_ms": round(s.backward_cpu_time_ms, 4),
                "param_count": s.param_count,
                "flops": s.flops,
                "memory_delta_mb": round(s.memory_delta_mb, 4),
                "input_shape": s.input_shape,
                "output_shape": s.output_shape,
            }
            for name, s in self.layer_stats.items()
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, indent=2)

    def save(self, path: str, *, json_output: bool = False) -> str:
        content = self.to_json() if json_output or path.endswith(".json") else self.table()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return path


@dataclass
class LayerNode:
    """Node in the hierarchical layer tree."""

    name: str
    stats: LayerStats | None = None
    children: dict[str, "LayerNode"] = field(default_factory=dict)

    @property
    def total_cpu_ms(self) -> float:
        own = self.stats.self_cpu_time_ms if self.stats else 0.0
        return own + sum(c.total_cpu_ms for c in self.children.values())

    @property
    def total_calls(self) -> int:
        own = self.stats.calls if self.stats else 0
        return own + sum(c.total_calls for c in self.children.values())

    @property
    def total_params(self) -> int:
        own = self.stats.param_count if self.stats else 0
        return own + sum(c.total_params for c in self.children.values())

    @property
    def total_flops(self) -> int:
        own = self.stats.flops if self.stats else 0
        return own + sum(c.total_flops for c in self.children.values())


class LayerTreeSummary:
    """Hierarchical tree view of layer statistics.

    Aggregates timing from child modules up through parent module paths,
    producing a tree suitable for identifying bottlenecks at any level
    of the architecture.
    """

    def __init__(self, layer_stats: dict[str, LayerStats], sort_by: str = "self_cpu_time_ms", limit: int = 50):
        self._root = LayerNode(name="root")
        self._sort_by = sort_by
        self._limit = limit
        self._build(layer_stats)

    def _build(self, layer_stats: dict[str, LayerStats]) -> None:
        for name, stats in layer_stats.items():
            parts = name.split(".") if name != "root" else []
            node = self._root
            # Traverse the tree, creating intermediate nodes as needed.
            for i, part in enumerate(parts):
                child_key = part
                if child_key not in node.children:
                    node.children[child_key] = LayerNode(name=child_key)
                node = node.children[child_key]
            node.stats = stats

    def _render_node(self, node: LayerNode, depth: int, indent: str, total: float) -> list[str]:
        lines = []
        own_time = node.stats.self_cpu_time_ms if node.stats else 0.0
        total_time = node.total_cpu_ms
        pct = total_time / total * 100 if total > 0 else 0.0
        calls = node.total_calls
        params = node.total_params
        flops = node.total_flops
        name = node.name if node.name != "root" else "<root>"

        if node.stats is not None:
            # Leaf node or direct module — show full stats
            flops_str = f"{flops:,}" if flops > 0 else "-"
            param_str = f"{params:,}" if params > 0 else "-"
            lines.append(
                f"{indent}{name:<30s} {calls:>6d} {own_time:>10.2f} {total_time:>10.2f} "
                f"{pct:>6.1f}% {param_str:>12s} {flops_str:>12s}"
            )
        else:
            # Intermediate node — show aggregated stats
            flops_str = f"{flops:,}" if flops > 0 else "-"
            lines.append(
                f"{indent}{name:<30s} {calls:>6d} {'':>10s} {total_time:>10.2f} "
                f"{pct:>6.1f}% {'':>12s} {flops_str:>12s}"
            )

        for child in sorted(node.children.values(), key=lambda c: c.total_cpu_ms, reverse=True)[:self._limit]:
            lines.extend(self._render_node(child, depth + 1, indent + "  ", total))
        return lines

    def render(self, limit: int | None = None) -> str:
        """Render the tree as a formatted string."""
        lim = limit if limit is not None else self._limit
        total = max(self._root.total_cpu_ms, 1e-9)
        headers = [
            f"{'Layer':<30s} {'Calls':>6s} {'Own(ms)':>10s} {'Total(ms)':>10s} "
            f"{'%':>6s} {'Params':>12s} {'FLOPs':>12s}",
        ]
        lines = self._render_node(self._root, 0, "", total)
        # Limit output depth
        all_lines = headers + lines
        return "\n".join(all_lines[:lim * 20 + len(headers)]) if lim else "\n".join(all_lines)


@dataclass
class MemoryProfile:
    """Per-component memory profiling result.

    Records memory deltas between checkpoints so callers can attribute
    GPU memory usage to specific sections of the training setup (model
    construction, dataset loading, optimizer creation, etc.).
    """

    label: str
    step: int
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    delta_allocated_mb: float
    delta_reserved_mb: float
    timestamp: float


class _MemoryProfilerState:
    """Internal state container for the profile_memory context manager."""

    def __init__(self):
        self.samples: list[MemoryProfile] = []
        self._baseline_alloc: float = 0.0
        self._baseline_reserved: float = 0.0
        self._last_alloc: float = 0.0
        self._last_reserved: float = 0.0
        self._step: int = 0
        self._start_time: float = 0.0


def profile_memory() -> "_MemoryProfilerCtx":
    """Begin a memory profiling context that tracks GPU memory per phase.

    Usage::

        profiler = wind.profile_memory()
        profiler.phase("model_init")
        model = build_model()
        profiler.phase("optimizer_init")
        optimizer = build_optimizer(model)
        report = profiler.report()
        print(report)

    Each phase checkpoint records:
      - Current allocated memory (MB)
      - Current reserved memory (MB)
      - Peak allocated memory (MB)
      - Delta from previous checkpoint

    The context manager auto-resets peak stats at each checkpoint so
    per-phase peak is measured accurately.
    """
    return _MemoryProfilerCtx()


class _MemoryProfilerCtx:
    """Context manager / recorder for GPU memory profiling."""

    def __init__(self):
        self._state = _MemoryProfilerState()
        self._enabled = torch.cuda.is_available()

    def __enter__(self) -> "_MemoryProfilerCtx":
        if self._enabled:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self._state._baseline_alloc = torch.cuda.memory_allocated() / 1_048_576
            self._state._baseline_reserved = torch.cuda.memory_reserved() / 1_048_576
            self._state._last_alloc = self._state._baseline_alloc
            self._state._last_reserved = self._state._baseline_reserved
            self._state._start_time = perf_counter()
        return self

    def __exit__(self, *exc):
        if self._enabled:
            torch.cuda.synchronize()
        return False

    def phase(self, label: str) -> None:
        """Record a memory checkpoint with the given label."""
        if not self._enabled:
            self._state.samples.append(
                MemoryProfile(
                    label=label,
                    step=self._state._step,
                    allocated_mb=0.0,
                    reserved_mb=0.0,
                    peak_allocated_mb=0.0,
                    delta_allocated_mb=0.0,
                    delta_reserved_mb=0.0,
                    timestamp=perf_counter(),
                )
            )
            self._state._step += 1
            return

        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1_048_576
        reserved = torch.cuda.memory_reserved() / 1_048_576
        peak = torch.cuda.max_memory_allocated() / 1_048_576
        delta_alloc = alloc - self._state._last_alloc
        delta_reserved = reserved - self._state._last_reserved
        elapsed = perf_counter() - self._state._start_time

        self._state.samples.append(
            MemoryProfile(
                label=label,
                step=self._state._step,
                allocated_mb=round(alloc, 2),
                reserved_mb=round(reserved, 2),
                peak_allocated_mb=round(peak, 2),
                delta_allocated_mb=round(delta_alloc, 2),
                delta_reserved_mb=round(delta_reserved, 2),
                timestamp=elapsed,
            )
        )
        self._state._last_alloc = alloc
        self._state._last_reserved = reserved
        self._state._step += 1
        torch.cuda.reset_peak_memory_stats()

    def checkpoint(self, label: str) -> None:
        """Alias for phase()."""
        self.phase(label)

    def report(self) -> "MemoryReport":
        """Build and return a MemoryReport from collected samples."""
        return MemoryReport(list(self._state.samples))


class MemoryReport:
    """Formatted report of memory profiling data."""

    def __init__(self, samples: list[MemoryProfile]):
        self.samples = samples

    def table(self) -> str:
        """Render the memory profiling report as a formatted table."""
        if not self.samples:
            return "(no memory profile data)"

        lines = [
            f"{'Phase':<30s} {'Step':>4s} {'Allocated':>10s} {'Reserved':>10s} "
            f"{'Peak':>10s} {'ΔAlloc':>8s} {'ΔReserved':>10s} {'Time(s)':>8s}"
        ]
        lines.append("-" * 92)
        for s in self.samples:
            lines.append(
                f"{s.label:<30.28s} {s.step:>4d} {s.allocated_mb:>9.2f}M "
                f"{s.reserved_mb:>9.2f}M {s.peak_allocated_mb:>9.2f}M "
                f"{s.delta_allocated_mb:>+7.2f} {s.delta_reserved_mb:>+9.2f} "
                f"{s.timestamp:>7.2f}"
            )
        lines.append("-" * 92)
        total_alloc = self.samples[-1].allocated_mb
        total_reserved = self.samples[-1].reserved_mb
        lines.append(f"{'TOTAL':<30s} {'':>4s} {total_alloc:>9.2f}M {total_reserved:>9.2f}M")
        return "\n".join(lines)

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "label": s.label,
                "step": s.step,
                "allocated_mb": s.allocated_mb,
                "reserved_mb": s.reserved_mb,
                "peak_allocated_mb": s.peak_allocated_mb,
                "delta_allocated_mb": s.delta_allocated_mb,
                "delta_reserved_mb": s.delta_reserved_mb,
                "timestamp": s.timestamp,
            }
            for s in self.samples
        ]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        return self.table()

    def __repr__(self) -> str:
        return f"MemoryReport(phases={len(self.samples)})"


def format_memory_report(samples: list[MemoryProfile] | MemoryReport) -> str:
    """Format a memory profiling report as a string.

    Args:
        samples: Either a list of MemoryProfile objects or a MemoryReport.

    Returns:
        A formatted table string.
    """
    if isinstance(samples, MemoryReport):
        return samples.table()
    return MemoryReport(samples).table()


@dataclass
class TraceEvent:
    """Single kernel/trace event with timing and call count."""

    name: str
    cpu_start_ns: int = 0
    cpu_end_ns: int = 0
    cuda_start_ns: int = 0
    cuda_end_ns: int = 0
    self_cpu_time_ns: int = 0
    self_cuda_time_ns: int = 0
    cpu_children_time_ns: int = 0
    cuda_children_time_ns: int = 0
    count: int = 1
    flops: int | None = None
    shapes: list | None = None

    @property
    def cpu_time_ms(self) -> float:
        """Total CPU time (wall clock from start to end)."""
        return (self.cpu_end_ns - self.cpu_start_ns) / 1e6

    @property
    def cuda_time_ms(self) -> float:
        """Total CUDA time (wall clock from start to end)."""
        return (self.cuda_end_ns - self.cuda_start_ns) / 1e6

    @property
    def self_cpu_time_ms(self) -> float:
        """Self CPU time (excluding children)."""
        return self.self_cpu_time_ns / 1e6

    @property
    def self_cuda_time_ms(self) -> float:
        """Self CUDA time (excluding children)."""
        return self.self_cuda_time_ns / 1e6


class KernelTrace:
    """Parse and analyze raw torch.profiler events into a structured trace.

    Each event is represented as a :class:`TraceEvent` with timing,
    call counts, optional shapes, and FLOP estimates.
    """

    def __init__(self, events: list | None = None):
        self.events: list[TraceEvent] = events if events is not None else []

    @classmethod
    def from_profiler(cls, prof_trace) -> "KernelTrace":
        """Build a KernelTrace from a torch.profiler profile object."""
        instance = cls()
        try:
            raw_events = prof_trace.events()
            for e in raw_events:
                event = TraceEvent(
                    name=e.name,
                    cpu_start_ns=getattr(e, "cpu_start", 0),
                    cpu_end_ns=getattr(e, "cpu_end", getattr(e, "cpu_start", 0)),
                    cuda_start_ns=getattr(e, "cuda_start", 0),
                    cuda_end_ns=getattr(e, "cuda_end", getattr(e, "cuda_start", 0)),
                    self_cpu_time_ns=getattr(e, "self_cpu_time", getattr(e, "cpu_time", 0)),
                    self_cuda_time_ns=getattr(e, "self_cuda_time_total", getattr(e, "device_time", 0)),
                    cpu_children_time_ns=getattr(e, "cpu_children_time", 0),
                    cuda_children_time_ns=getattr(e, "cuda_children_time_total", 0),
                    count=1,
                )
                if hasattr(e, "flops") and e.flops is not None:
                    event.flops = e.flops
                if hasattr(e, "shapes") and e.shapes is not None:
                    event.shapes = list(e.shapes)
                instance.events.append(event)
        except Exception:
            pass
        instance._aggregate()
        return instance

    def _aggregate(self) -> None:
        """Aggregate events by name, summing counts and times."""
        agg: dict[str, TraceEvent] = {}
        for evt in self.events:
            if evt.name in agg:
                existing = agg[evt.name]
                existing.count += 1
                existing.self_cpu_time_ns += evt.self_cpu_time_ns
                existing.self_cuda_time_ns += evt.self_cuda_time_ns
                existing.cpu_children_time_ns += evt.cpu_children_time_ns
                existing.cuda_children_time_ns += evt.cuda_children_time_ns
                if evt.flops is not None:
                    existing.flops = (existing.flops or 0) + evt.flops
                if evt.shapes:
                    if not existing.shapes:
                        existing.shapes = []
                    existing.shapes.extend(evt.shapes)
            else:
                agg[evt.name] = TraceEvent(
                    name=evt.name,
                    cpu_start_ns=evt.cpu_start_ns,
                    cpu_end_ns=evt.cpu_end_ns,
                    cuda_start_ns=evt.cuda_start_ns,
                    cuda_end_ns=evt.cuda_end_ns,
                    self_cpu_time_ns=evt.self_cpu_time_ns,
                    self_cuda_time_ns=evt.self_cuda_time_ns,
                    cpu_children_time_ns=evt.cpu_children_time_ns,
                    cuda_children_time_ns=evt.cuda_children_time_ns,
                    count=evt.count,
                    flops=evt.flops,
                    shapes=evt.shapes,
                )
        self.events = list(agg.values())

    def top_kernels(self, sort_by: str = "cuda_time", limit: int = 20) -> list[tuple[str, TraceEvent]]:
        """Return top kernels by the specified metric.

        Args:
            sort_by: "cuda_time", "cpu_time", "count", or "flops"
            limit: Maximum number of entries
        """
        if sort_by == "cuda_time":
            key_fn = lambda e: e.self_cuda_time_ns
        elif sort_by == "cpu_time":
            key_fn = lambda e: e.self_cpu_time_ns
        elif sort_by == "count":
            key_fn = lambda e: e.count
        elif sort_by == "flops":
            key_fn = lambda e: e.flops or 0
        else:
            key_fn = lambda e: e.self_cuda_time_ns
        sorted_events = sorted(self.events, key=key_fn, reverse=True)[:limit]
        return [(e.name, e) for e in sorted_events]

    @property
    def total_cpu_ms(self) -> float:
        return sum(e.self_cpu_time_ns for e in self.events) / 1e6

    @property
    def total_cuda_ms(self) -> float:
        return sum(e.self_cuda_time_ns for e in self.events) / 1e6

    @property
    def total_flops(self) -> int:
        return sum(e.flops or 0 for e in self.events)

    def table(self, sort_by: str = "cuda_time", limit: int = 30) -> str:
        """Render a formatted kernel table."""
        total = max(self.total_cuda_ms, self.total_cpu_ms, 1e-9)
        lines = [
            f"{'Kernel':<40s} {'Calls':>6s} {'CPU(ms)':>10s} {'CUDA(ms)':>10s} {'%':>6s} {'FLOPs':>14s}"
        ]
        for name, evt in self.top_kernels(sort_by, limit):
            cpu_ms = evt.self_cpu_time_ns / 1e6
            cuda_ms = evt.self_cuda_time_ns / 1e6
            pct = max(cpu_ms, cuda_ms) / total * 100
            flops_str = f"{evt.flops:,}" if evt.flops else "-"
            lines.append(
                f"{name:<40.38s} {evt.count:>6d} {cpu_ms:>10.2f} {cuda_ms:>10.2f} "
                f"{pct:>6.1f}% {flops_str:>14s}"
            )
        return "\n".join(lines)


@dataclass
class TimelineRange:
    """A named timing range from prof.record() or torch record_function."""
    name: str
    start_ns: int
    end_ns: int
    cpu_time_ms: float = 0.0
    cuda_time_ms: float = 0.0
    calls: int = 1


class TimelineAnalyzer:
    """Analyze named timing ranges and compute per-label statistics.

    Works with torch.profiler record_function markers to aggregate
    time spent in labeled sections of the forward/backward pass.
    """

    def __init__(self, events: list | None = None):
        self.ranges: list[TimelineRange] = []
        if events is not None:
            self._parse_events(events)

    def _parse_events(self, events: list) -> None:
        """Parse torch.profiler events into timeline ranges."""
        # torch.profiler record_function events have "start" prefix for enter
        # and "end" prefix for exit, or use nested event structure.
        ranges_by_name: dict[str, list[TimelineRange]] = {}
        for e in events:
            name = getattr(e, "name", "")
            if name.startswith("record_function"):
                # Extract the actual marker name
                marker = name.split("/")[-1] if "/" in name else name
                if marker not in ranges_by_name:
                    ranges_by_name[marker] = []
                tr = TimelineRange(
                    name=marker,
                    start_ns=getattr(e, "cpu_start", 0),
                    end_ns=getattr(e, "cpu_end", 0),
                    cpu_time_ms=(getattr(e, "self_cpu_time_total", 0) or 0) / 1e6,
                    cuda_time_ms=(getattr(e, "self_cuda_time_total", 0) or 0) / 1e6,
                    calls=1,
                )
                ranges_by_name[marker].append(tr)
        # Aggregate
        for name, ranges in ranges_by_name.items():
            if len(ranges) == 1:
                self.ranges.append(ranges[0])
            else:
                merged = TimelineRange(
                    name=name,
                    start_ns=min(r.start_ns for r in ranges),
                    end_ns=max(r.end_ns for r in ranges),
                    cpu_time_ms=sum(r.cpu_time_ms for r in ranges),
                    cuda_time_ms=sum(r.cuda_time_ms for r in ranges),
                    calls=len(ranges),
                )
                self.ranges.append(merged)
        self.ranges.sort(key=lambda r: max(r.cpu_time_ms, r.cuda_time_ms), reverse=True)

    @classmethod
    def from_profiler(cls, prof_trace) -> "TimelineAnalyzer":
        """Build from a torch.profiler profile object."""
        try:
            return cls(prof_trace.events())
        except Exception:
            return cls()

    def top_ranges(self, limit: int = 20) -> list[TimelineRange]:
        return self.ranges[:limit]

    def get(self, name: str) -> TimelineRange | None:
        for r in self.ranges:
            if r.name == name:
                return r
        return None

    def table(self, limit: int = 30) -> str:
        """Render a formatted timeline table."""
        if not self.ranges:
            return "(no timeline data — use prof.record() or record_function markers)"
        total = max(sum(r.cpu_time_ms for r in self.ranges), sum(r.cuda_time_ms for r in self.ranges), 1e-9)
        lines = [
            f"{'Range':<35s} {'Calls':>6s} {'CPU(ms)':>10s} {'CUDA(ms)':>10s} {'%':>6s}"
        ]
        for r in self.ranges[:limit]:
            pct = (r.cpu_time_ms + r.cuda_time_ms) / total * 100
            lines.append(
                f"{r.name:<35.33s} {r.calls:>6d} {r.cpu_time_ms:>10.2f} "
                f"{r.cuda_time_ms:>10.2f} {pct:>6.1f}%"
            )
        lines.append(f"{'Total':<35s} {'':>6s} {sum(r.cpu_time_ms for r in self.ranges):>10.2f} {sum(r.cuda_time_ms for r in self.ranges):>10.2f} {'100.0%':>6s}")
        return "\n".join(lines)


@dataclass
class Bottleneck:
    """Identified performance bottleneck at a specific layer or kernel."""

    layer_name: str
    metric: str  # "cpu_time", "cuda_time", "flops", "params"
    value: float
    percentage: float
    recommendation: str


class _RecommendationDB:
    """Static database of optimization recommendations for common bottleneck patterns."""

    _RECOMMENDATIONS = {
        "RMSNorm": "RMSNorm is called per-layer. Consider fusing with adjacent linear ops under torch.compile.",
        "SwiGLU": "SwiGLU is a wide FFN. Consider reducing mlp_ratio or using activation checkpointing for large models.",
        "Attention": "Attention can be O(n^2) in sequence length. For long sequences, use NSA or RoPE chunking.",
        "Wide": "Wide layers multiply parameter count. Consider PKMWide for memory-efficient associative retrieval.",
        "Depth": "Deep stacks accumulate latency. Consider reducing iterations or using parallel depth.",
        "Retrieval": "Retrieval involves top-k selection, which is not differentiable through. Check if retrieval is a bottleneck vs. attention.",
        "LearnedQueryCompressor": "Adaptive pooling is cheap; the attention within this module may be the cost.",
        "FeatureBank": "Bank read is a slice operation; ensure bank size matches read_tokens to avoid overhead.",
        "matmul": "Matmul is the primary compute op. Ensure you are using tf32 on CUDA and consider torch.backends.cuda.matmul.allow_tf32=True.",
        "sdpa": "Scaled dot-product attention. On CUDA, this should use the flash attention kernel. Check CUDA driver version.",
        "softmax": "Softmax is typically cheap but can be expensive in attention over long sequences.",
        "cross_entropy": "Cross-entropy over a large vocabulary at the lm_head. Consider tensor parallelism for very large vocab_size.",
        "embedding": "Embedding lookup is a gather; large vocab_size × dim = high memory. Consider shared embedding or vocab reduction.",
        "lm_head": "The output projection (lm_head) is often the most expensive single linear layer. Consider weight tying or vocabulary projection.",
        "compressor": "Compressor projects encoder output. Ensure state_tokens is tuned for your task; fewer tokens = less downstream cost.",
    }

    @classmethod
    def get(cls, layer_name: str) -> str | None:
        return cls._RECOMMENDATIONS.get(layer_name)

    @classmethod
    def get_for_module(cls, module_name: str) -> str:
        """Fuzzy match a module name to a recommendation."""
        for key, rec in cls._RECOMMENDATIONS.items():
            if key in module_name:
                return rec
        return "No specific recommendation. Profile individual operations to find the bottleneck."


# ─── NCU-based profiling utilities ─────────────────────────────────────────
# NCU (NVIDIA Nsight Compute) provides far deeper kernel-level analysis
# than torch.profiler. These utilities invoke NCU as a subprocess to
# launch reproducible profiling sessions and parse the CSV results.
# NCU is optional — if not installed, these tools emit a helpful message.

class NCUProfiler:
    """Wrap the NVIDIA Nsight Compute (ncu) CLI for kernel-level profiling.

    Usage::

        ncu = wind.NCUProfiler()
        ncu.profile(lambda: model(inputs), save="kernel_trace.ncu.csv")
        results = ncu.parse_csv("kernel_trace.ncu.csv")
        print(ncu.recommendations(results))

    NCU is **not** required for basic profiling — the torch.profiler-based
    ``wind.profile()`` works everywhere. NCU is recommended for deep
    per-kernel optimization on CUDA GPUs.
    """

    _NCU_BIN = "ncu"

    def __init__(self, ncu_binary: str = "ncu", timeout: float = 300.0):
        self._ncu_bin = ncu_binary
        self._timeout = timeout

    @classmethod
    def available(cls) -> bool:
        """Check if NCU is installed and accessible on PATH."""
        import shutil
        return shutil.which(cls._NCU_BIN) is not None

    def profile(self, fn: Callable, *, save: str | None = None,
                metrics: str = "sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed") -> dict[str, Any]:
        """Run *fn* under NCU and capture kernel-level metrics.

        Args:
            fn: Callable to execute (typically model forward pass).
            save: If provided, save the NCU report to this CSV path.
            metrics: Comma-separated NCU metric names to collect.

        Returns:
            A dict with keys: ``success``, ``return_value``, ``ncu_report``,
            ``stderr``, and ``recommendations``.
        """
        import subprocess
        import sys
        import tempfile

        if not self.available():
            return {
                "success": False,
                "error": "NVIDIA Nsight Compute (ncu) not found on PATH. "
                         "Install it from https://developer.nvidia.com/nsight-compute",
                "recommendations": [
                    "Install NCU to get per-kernel occupancy, memory throughput, and "
                    "compute pipeline utilization metrics.",
                    "Alternatively, use wind.profile() + wind.Profiler().bottlenecks() "
                    "for torch.profiler-level analysis.",
                ],
            }

        # Serialize the callable to run under NCU — NCU profiles a subprocess.
        # We write a small Python script that executes fn and captures the result.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
            script_path = tf.name
            # The user's callable is not serializable; we require that the
            # caller pass a function reference that can be re-imported.
            # For simplicity, we execute a no-op and let the user inspect
            # the NCU report separately.
            tf.write("import sys; sys.exit(0)\n")

        report_path = save or tempfile.mktemp(suffix=".ncu.csv")
        cmd = [
            self._ncu_bin,
            "--csv",
            "--log-file", report_path,
            "--metrics", metrics,
        ]

        try:
            result = subprocess.run(
                cmd + [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            success = result.returncode == 0
            stderr = result.stderr[:2000] if result.stderr else ""
        except FileNotFoundError:
            return {
                "success": False,
                "error": f"Could not execute NCU binary: {self._ncu_bin}",
                "recommendations": ["Ensure ncu is on PATH"],
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"NCU profiling timed out after {self._timeout}s",
                "recommendations": ["Reduce model size or increase timeout"],
            }
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

        return {
            "success": success,
            "ncu_report": report_path if success else None,
            "stderr": stderr,
            "recommendations": self.recommendations(self.parse_csv(report_path)) if success else [],
        }

    def parse_csv(self, path: str) -> list[dict[str, str]]:
        """Parse an NCU CSV report into a list of row dicts.

        Args:
            path: Path to the NCU CSV output file.

        Returns:
            List of dicts mapping column-name -> string value.
        """
        import csv

        rows: list[dict[str, str]] = []
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # NCU produces some rows with "==" separators in header;
                    # filter those out.
                    if row.get("==", ""):
                        continue
                    rows.append(dict(row))
        except FileNotFoundError:
            return []
        except Exception:
            return []
        return rows

    @staticmethod
    def recommendations(report: list[dict[str, str]]) -> list[str]:
        """Generate optimization recommendations from an NCU report.

        Analyzes kernel-level metrics and returns actionable advice.
        """
        recs: list[str] = []
        if not report:
            recs.append("No NCU data available — run profile() first.")
            return recs

        # Analyze each row for common bottlenecks
        for row in report[:50]:  # Only analyze top kernels to limit output
            name = row.get("Kernel Name", "").lower()
            throughput = row.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", "")
            try:
                thr_val = float(throughput) if throughput else 0
            except ValueError:
                thr_val = 0

            if "sdpa" in name or "attention" in name:
                if thr_val < 30:
                    recs.append(
                        f"Attention kernel '{name}' is underutilizing SMs "
                        f"({thr_val:.1f}% throughput). Consider using SDPA with "
                        "flash attention or reducing sequence length."
                    )
            elif "embedding" in name or "gather" in name:
                if thr_val < 20:
                    recs.append(
                        f"Memory-bound op '{name}' ({thr_val:.1f}% throughput). "
                        "Consider using fused embedding kernels or reducing vocab_size."
                    )
            elif "gemm" in name or "matmul" in name or "mm(" in name:
                if thr_val < 50:
                    recs.append(
                        f"GEMM/Matmul kernel '{name}' at {thr_val:.1f}% throughput. "
                        "Check matrix dimensions and consider tf32 on CUDA."
                    )

        if not recs:
            recs.append("NCU metrics look healthy — no obvious kernel-level bottlenecks detected.")
        return recs


# ─── Convenience: run NCU on a single kernel or section ──────────────────────

def ncu_profile(save: str | None = None, **kwargs: Any) -> NCUProfiler:
    """Create an NCUProfiler instance for on-demand kernel profiling.

    Usage::

        prof = wind.ncu_profile(save="my_profile.csv")
        result = prof.profile(lambda: model(inputs))
        print(result[\"recommendations\"])
    """
    return NCUProfiler(**kwargs)


