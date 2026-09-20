"""Minimal metrics logging interface.

Loggers are plain callables ``metrics: dict[str, float] -> None``.
Built-in options: ``"console"`` (human-readable), ``"json"`` (line-delimited JSON).
Pass ``logger=None`` to silence logging entirely.
"""

from __future__ import annotations

import json as _json
import sys
from typing import Any, Callable

Logger = Callable[[dict[str, Any]], None] | None


def console_logger(stream: Any = sys.stderr) -> Logger:
    """Render a compact one-line table of metrics to *stream*."""
    def _log(metrics: dict[str, Any]) -> None:
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
        stream.write("\t".join(parts) + "\n")
        stream.flush()
    return _log


def json_logger(stream: Any = sys.stdout) -> Logger:
    """Emit one JSON object per line with a monotonic timestamp."""
    import time as _time
    def _log(metrics: dict[str, Any]) -> None:
        payload = {"ts": _time.time(), **metrics}
        stream.write(_json.dumps(payload) + "\n")
        stream.flush()
    return _log


def make_logger(kind: str | Logger | None) -> Logger:
    """Resolve a logger specifier to a callable (or None)."""
    if kind is None:
        return None
    if callable(kind) and not isinstance(kind, str):
        return kind
    if kind == "console":
        return console_logger()
    if kind == "json":
        return json_logger()
    raise ValueError(f"unknown logger kind: {kind!r}")
