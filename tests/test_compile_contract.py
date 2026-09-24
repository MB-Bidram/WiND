"""Compile behavior contracts that do not hide eager fallbacks."""

import pytest
import torch

from winc._internal.guard import GUARD_ENABLED
from winc.modules import RMSNorm, SwiGLU, Wide
from winc.runtime import HardwareProfile, OptimizedBackend


def test_runtime_compile_failure_is_not_silently_eager(monkeypatch):
    """A requested compilation must surface its failure to the caller."""
    module = torch.nn.Linear(4, 4)

    def fail_compile(_):
        raise RuntimeError("intentional compile failure")

    monkeypatch.setattr(torch, "compile", fail_compile)
    backend = OptimizedBackend(HardwareProfile(torch.device("cpu"), torch.float32, compile=True))
    with pytest.raises(RuntimeError, match="not silently downgraded"):
        backend.prepare(module)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_wide_fullgraph_compile_when_runtime_guards_are_disabled():
    """The normal execution policy is full-graph capturable for a small Wide."""
    if GUARD_ENABLED:
        pytest.skip("WIND_GUARDS=1 intentionally enables eager diagnostic scans")
    model = Wide(
        torch.nn.Sequential(RMSNorm(8), SwiGLU(8, 16)),
        torch.nn.Sequential(RMSNorm(8), SwiGLU(8, 16)),
        dim=8,
        mode="sum",
    ).eval()
    x = torch.randn(2, 4, 8)
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), model(x))
