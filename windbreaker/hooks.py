"""Hook-only execution inspection for generic WiND and LanguageModel.

This module intentionally identifies models by their public structure instead
of importing WiND internals.  It therefore adds no reverse dependency from
WiND to WiNDBreaker and does not patch forwards or model state.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn


@dataclass(frozen=True)
class BankProperties:
    """Independent Bank properties recorded on each read event."""

    immutable_during_recurrence: bool
    lifetime: str
    differentiable_end_to_end: bool


@dataclass(frozen=True)
class StateStats:
    shape: tuple[int, ...]
    dtype: str
    device: str
    numel: int
    requires_grad: bool
    mean: float | None = None
    std: float | None = None


@dataclass
class LayerApplication:
    reasoning_pass: int
    module: str
    state: StateStats
    gradient_seen: bool = False
    # Optional detached CPU state, captured only when explicitly requested.
    # This is intentionally absent from normal low-overhead inspection.
    snapshot: torch.Tensor | None = None


@dataclass(frozen=True)
class BankReadEvent:
    reasoning_pass: int
    module: str
    properties: BankProperties
    bank: StateStats


@dataclass(frozen=True)
class StageApplication:
    """One observed Wide/Encoder execution used by causal ablation reports."""
    module: str
    state: StateStats


@dataclass
class InspectionReport:
    path: Literal["generic", "language"]
    layer_applications: list[LayerApplication] = field(default_factory=list)
    bank_reads: list[BankReadEvent] = field(default_factory=list)
    stage_applications: list[StageApplication] = field(default_factory=list)


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _stats(tensor: torch.Tensor, summary: bool) -> StateStats:
    # Metadata-only mode does not issue a reduction or a host synchronization.
    mean = std = None
    if summary:
        detached = tensor.detach()
        mean = float(detached.float().mean().cpu())
        std = float(detached.float().std(unbiased=False).cpu())
    return StateStats(
        shape=tuple(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device),
        numel=tensor.numel(), requires_grad=tensor.requires_grad, mean=mean, std=std,
    )


class Inspection(AbstractContextManager):
    """Attach PyTorch hooks and collect a normalized WiND execution report.

    Generic WND emits one application per cyclic reasoning iteration.
    LanguageModel emits one application for each layer of each full-stack pass.
    Hooks only exist while this object is attached; disabled inspection adds no
    model-side work.
    """

    def __init__(self, model: nn.Module, *, state_stats: Literal["metadata", "summary"] = "metadata",
                 state_capture: Literal["none", "cpu"] = "none"):
        if state_stats not in {"metadata", "summary"}:
            raise ValueError("state_stats must be 'metadata' or 'summary'")
        if state_capture not in {"none", "cpu"}:
            raise ValueError("state_capture must be 'none' or 'cpu'")
        self.model = model
        self._summary = state_stats == "summary"
        self._capture = state_capture == "cpu"
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._generic = hasattr(getattr(model, "depth", None), "layers")
        self._depth_count = len(getattr(model.depth, "layers", ())) if self._generic else len(getattr(model, "depth", ()))
        if self._depth_count < 1:
            raise TypeError("model does not expose a supported WiND depth stack")
        self.report = InspectionReport(path="generic" if self._generic else "language")
        self._applications = 0
        self.attach()

    def __enter__(self) -> "Inspection":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.detach()

    def attach(self) -> "Inspection":
        if self._handles:
            return self
        named = dict(self.model.named_modules())
        if self._generic:
            for index in range(self._depth_count):
                name = f"depth.layers.{index}"
                self._handles.append(named[name].register_forward_hook(self._layer_hook(name)))
            retrieval = named.get("depth.retrieval")
            if retrieval is not None:
                self._handles.append(retrieval.register_forward_hook(self._generic_bank_hook, with_kwargs=True))
            for name in ("wide", "encoder"):
                if name in named: self._handles.append(named[name].register_forward_hook(self._stage_hook(name)))
        else:
            for index in range(self._depth_count):
                name = f"depth.{index}"
                self._handles.append(named[name].register_forward_hook(self._layer_hook(name)))
            for name in ("wide", *[f"encoder.{i}" for i in range(len(getattr(self.model, "encoder", ())) )]):
                if name in named: self._handles.append(named[name].register_forward_hook(self._stage_hook(name)))
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _properties(self) -> BankProperties:
        if self._generic:
            bank = getattr(self.model, "bank", None)
            return BankProperties(True, "request-local", not bool(getattr(bank, "detach", True)))
        return BankProperties(True, "request-local", True)

    def _stage_hook(self, name: str):
        def hook(module, inputs, output):
            state = _first_tensor(output)
            if state is not None:
                self.report.stage_applications.append(StageApplication(name, _stats(state, self._summary)))
        return hook

    def _layer_hook(self, name: str):
        def hook(module, inputs, output):
            state = _first_tensor(output)
            if state is None:
                return
            pass_index = self._applications if self._generic else self._applications // self._depth_count
            snapshot = state.detach().to(device="cpu", copy=True) if self._capture else None
            event = LayerApplication(pass_index, name, _stats(state, self._summary), snapshot=snapshot)
            self.report.layer_applications.append(event)
            self._applications += 1
            if state.requires_grad:
                state.register_hook(lambda grad, event=event: setattr(event, "gradient_seen", True))
            # In LanguageModel, a depth block's second positional input is the
            # request-local learned-query bank consumed by cross-attention.
            if not self._generic and len(inputs) > 1 and isinstance(inputs[1], torch.Tensor):
                self.report.bank_reads.append(BankReadEvent(
                    event.reasoning_pass, name, self._properties(), _stats(inputs[1], self._summary)
                ))
        return hook

    def _generic_bank_hook(self, module, inputs, kwargs, output):
        # Retrieval receives a pre-projected cache on recurrence.  Its input
        # state is the consumer; cache tensors are the bounded bank read.
        cache = _first_tensor(kwargs.get("bank_cache"))
        if cache is None and len(inputs) > 1:
            cache = _first_tensor(inputs[1])
        if cache is None:
            cache = _first_tensor(kwargs.get("bank"))
        if cache is not None:
            self.report.bank_reads.append(BankReadEvent(
                len(self.report.bank_reads), "depth.retrieval", self._properties(), _stats(cache, self._summary)
            ))


def inspect_model(model: nn.Module, *, state_stats: Literal["metadata", "summary"] = "metadata",
                  state_capture: Literal["none", "cpu"] = "none") -> Inspection:
    """Attach a normalized architecture inspector; use as a context manager."""

    return Inspection(model, state_stats=state_stats, state_capture=state_capture)
