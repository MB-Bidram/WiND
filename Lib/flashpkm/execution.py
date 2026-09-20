"""Explicit high-performance execution policies for FlashPKM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


RetrievalBackend = Literal["auto", "torch", "triton"]
FactorTopKBackend = Literal["auto", "torch", "hierarchical"]


@dataclass(frozen=True)
class PKMInferenceConfig:
    """Configure a reproducible inference execution path.

    ``torch`` retains standard PyTorch retrieval operations. ``triton`` uses
    the exact F=2 fused retrieval kernel when eligible. ``auto`` selects that
    kernel only for no-grad F=2 CUDA inference and otherwise uses PyTorch.
    """

    dtype: torch.dtype | None = None
    retrieval_backend: RetrievalBackend = "auto"
    factor_topk_backend: FactorTopKBackend = "auto"
    compile_model: bool = False

    def __post_init__(self) -> None:
        if self.retrieval_backend not in {"auto", "torch", "triton"}:
            raise ValueError(f"unknown retrieval backend: {self.retrieval_backend!r}")
        if self.factor_topk_backend not in {"auto", "torch", "hierarchical"}:
            raise ValueError(f"unknown factor top-k backend: {self.factor_topk_backend!r}")
        if self.dtype not in {None, torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError("dtype must be None, float32, float16, or bfloat16")


class PKMInferenceEngine:
    """Prepare and execute a PKM module with an explicit backend policy."""

    def __init__(self, module: nn.Module, config: PKMInferenceConfig | None = None):
        self.config = config or PKMInferenceConfig()
        self.module = module
        prepare = getattr(module, "prepare_for_inference", None)
        if prepare is None:
            module.eval()
            if self.config.dtype is not None:
                module.to(dtype=self.config.dtype)
        else:
            try:
                prepare(
                    self.config.dtype,
                    self.config.retrieval_backend,
                    self.config.factor_topk_backend,
                )
            except TypeError:
                prepare(self.config.dtype)

        self._callable = module
        if self.config.compile_model:
            self._callable = torch.compile(module, fullgraph=True)

    @torch.inference_mode()
    def __call__(self, query: torch.Tensor):
        if self.config.dtype is not None and query.dtype != self.config.dtype:
            raise ValueError(
                f"expected input dtype {self.config.dtype}, got {query.dtype}; "
                "cast inputs before invoking the inference engine"
            )
        return self._callable(query)
