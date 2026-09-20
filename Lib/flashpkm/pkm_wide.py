"""Product Key Memory based Wide stage for WiND.

PKMWide is a drop-in alternative to Wide that uses associative memory
retrieval instead of dense neural layers. It wraps a small QueryEncoder
that maps input representations into a retrieval-friendly query space,
feeding into a FactorizedPKM for memory lookup.

Architecture:
  Input [B, S, dim] -> input_norm -> QueryEncoder -> query_norm
  -> FactorizedPKM (retrieval) -> value_norm -> output_proj
  -> output_norm -> gated/residual integration with input
"""

from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .modules import WindModule
from .pkm import FactorizedPKM
from .pkm_utils import QueryEncoder, _resolve_norm


def _profile_range(name: str):
    if torch.compiler.is_compiling():
        return nullcontext()
    if torch.autograd._profiler_enabled():
        return torch.profiler.record_function(name)
    return nullcontext()


class PKMWide(WindModule):
    """Product Key Memory based Wide stage.

    A drop-in replacement for :class:`~wind.engine.modules.Wide` that uses
    factorized associative memory retrieval. PKM is a memory system, not a
    universal feature extractor — a small learned query encoder maps incoming
    representations into a retrieval-friendly query space.

    Parameters
    ----------
    dim : int
        Input/output (model) dimension.
    memory_size : int
        Number of slots per sub-key table.
    effective_memory_size : int or None
        Target Cartesian product-key address capacity. It derives the
        per-factor table size and is opt-in; memory_size continues to mean
        slots per factor for backward compatibility.
    query_dim : int
        Dimensionality of the query representation. Must be divisible by
        ``num_factors * head_dim``.
    value_dim : int
        Dimensionality of stored value vectors. Can differ from ``dim``;
        the output projection maps back when needed.
    num_factors : int
        Number of factorized sub-key tables.
    heads : int
        Number of retrieval heads. Each head has its own sub-queries.
    topk : int
        Number of final candidates to retrieve globally per head.
    topk_per_factor : int or None
        Per-factor top-k before Cartesian-product scoring.
    head_dim : int or None
        Dimension of each sub-query within a head. Defaults to
        ``query_dim // (heads * num_factors)``.
    subkey_dim : int or None
        Dimension of each sub-key lookup vector.
    similarity : str
        "dot" or "cosine" scoring.
    scale_factor : float or None
        Score scale for dot-product.
    query_encoder_depth : int
        Number of layers in the query encoder MLP.
    query_encoder_hidden_dim : int or None
        Hidden dimension of intermediate encoder layers.
    query_activation : str
        Activation for intermediate encoder layers.
    query_norm : str or None
        Normalization for the query encoder intermediate layers.
    query_dropout : float
        Dropout in the query encoder.
    query_residual : bool
        Residual connections in the query encoder.
    query_bias : bool
        Bias in query encoder linear layers.

    Norm options (each independently configurable):
    input_norm, query_norm_retrieval, key_norm, value_norm, output_norm
    """

    def __init__(
        self,
        dim: int,
        memory_size: int = 4096,
        query_dim: int = 128,
        value_dim: int = 256,
        num_factors: int = 2,
        heads: int = 4,
        topk: int = 32,
        topk_per_factor: int | None = None,
        head_dim: int | None = None,
        subkey_dim: int | None = None,
        similarity: str = "cosine",
        scale_factor: float | None = None,
        query_encoder_depth: int = 1,
        query_encoder_hidden_dim: int | None = None,
        query_activation: str = "silu",
        query_encoder_norm: str | None = None,
        query_dropout: float = 0.0,
        query_residual: bool = False,
        query_bias: bool = False,
        input_norm: str | None = "rmsnorm",
        query_norm_retrieval: str | None = "rmsnorm",
        key_norm: str | None = "none",
        value_norm: str | None = "none",
        output_norm: str | None = "rmsnorm",
        output_mode: str = "residual",
        gate_init: float = 0.0,
        key_dtype: torch.dtype = torch.float32,
        value_dtype: torch.dtype = torch.float32,
        effective_memory_size: int | None = None,
        exact_candidate_pruning: bool = False,
    ):
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        if memory_size < 1:
            raise ValueError("memory_size must be positive")
        if effective_memory_size is not None and effective_memory_size < 1:
            raise ValueError("effective_memory_size must be positive")
        if query_dim < 1:
            raise ValueError("query_dim must be positive")
        if value_dim < 1:
            raise ValueError("value_dim must be positive")
        if num_factors < 1:
            raise ValueError("num_factors must be at least 1")
        if heads < 1:
            raise ValueError("heads must be positive")
        if topk < 1:
            raise ValueError("topk must be positive")
        if similarity not in {"dot", "cosine"}:
            raise ValueError(f"unknown similarity: {similarity!r}")
        if output_mode not in {"none", "residual", "gated"}:
            raise ValueError(f"unknown output_mode: {output_mode!r}")

        # Validate query_dim divisibility across heads and factors
        if head_dim is None:
            head_dim = query_dim // (heads * num_factors)
        if query_dim != head_dim * heads * num_factors:
            raise ValueError(
                f"query_dim ({query_dim}) must equal head_dim*heads*num_factors "
                f"({head_dim}*{heads}*{num_factors}={head_dim * heads * num_factors})"
            )

        # Each head gets query_dim//heads dimensional sub-queries
        per_head_query_dim = query_dim // heads
        per_head_subkey_dim = head_dim * num_factors

        # Validate per-head divisibility
        if per_head_query_dim != head_dim * num_factors:
            raise ValueError(
                f"query_dim/head ({per_head_query_dim}) must equal head_dim*num_factors "
                f"({head_dim}*{num_factors})"
            )

        self.dim = dim
        self.memory_size = memory_size
        self.effective_memory_size = effective_memory_size
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.num_factors = num_factors
        self.heads = heads
        self.topk = topk
        self.head_dim = head_dim
        self.subkey_dim = subkey_dim or head_dim
        self.similarity = similarity
        self.output_mode = output_mode
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype

        # Input normalization
        self.input_norm_layer = _resolve_norm(input_norm, dim)

        # Query encoder: maps [B, S, dim] -> [B, S, query_dim]
        # The encoder output is reshaped for multi-head retrieval
        self.query_encoder = QueryEncoder(
            input_dim=dim,
            query_dim=query_dim,
            hidden_dim=query_encoder_hidden_dim or dim,
            depth=query_encoder_depth,
            activation=query_activation,
            norm=query_encoder_norm,
            dropout=query_dropout,
            residual=query_residual,
            bias=query_bias,
        )

        # Normalization applied after query encoder
        self.query_norm_layer = _resolve_norm(query_norm_retrieval, query_dim)

        # Single batched PKM instance handling all heads
        # keys: [heads, num_factors, memory_size, subkey_dim]
        # values: [heads, memory_size, num_factors, value_dim]
        self.pkm = FactorizedPKM(
            query_dim=query_dim,
            memory_size=memory_size,
            value_dim=value_dim,
            num_factors=num_factors,
            heads=heads,
            topk=topk,
            topk_per_factor=topk_per_factor,
            head_dim=head_dim,
            subkey_dim=subkey_dim or head_dim,
            similarity=similarity,
            scale_factor=scale_factor,
            key_init_norm=key_norm != "none",
            key_dtype=key_dtype,
            value_dtype=value_dtype,
            effective_memory_size=effective_memory_size,
            exact_candidate_pruning=exact_candidate_pruning,
        )
        self.memory_size = self.pkm.memory_size
        self.effective_memory_size = self.pkm.effective_memory_size

        # Key/value normalization (documented for configurability; the PKM
        # applies key normalization at init when key_norm != "none")
        self.key_norm_name = key_norm
        self.value_norm_name = value_norm
        self.value_norm_layer = _resolve_norm(value_norm, value_dim) if value_norm != "none" else None

        # Output projection: maps [B, S, heads*value_dim] -> [B, S, dim]
        self.output_proj = nn.Linear(heads * value_dim, dim, bias=False)

        # Output normalization
        self.output_norm_layer = _resolve_norm(output_norm, dim)

        # Gating for residual mode
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def prepare_for_inference(
        self,
        dtype: torch.dtype | None = None,
        retrieval_backend: str = "auto",
        factor_topk_backend: str = "auto",
    ) -> "PKMWide":
        super().prepare_for_inference(dtype)
        self.pkm.set_retrieval_backend(retrieval_backend)
        self.pkm.set_factor_topk_backend(factor_topk_backend)
        return self

    @torch.inference_mode()
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """Run the configured no-grad inference path."""
        return self(x)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_branches: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward pass: [B, S, dim] -> [B, S, dim]

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, S, dim].
        return_branches : bool
            If True, include per-head outputs in the return tuple (for
            regularization diagnostics).
        return_aux : bool
            If True, include retrieval diagnostics in the return.

        Returns
        -------
        output : [B, S, dim]
        """
        with _profile_range("PKMWide.forward"):
            original = x

            # Input normalization
            with _profile_range("PKMWide.input_normalization"):
                if self.input_norm_layer is not None:
                    x = self.input_norm_layer(x)

            # Query encoding: [B, S, dim] -> [B, S, query_dim]
            with _profile_range("PKMWide.score_projection"):
                x = self.query_encoder(x)

            # Query normalization
            with _profile_range("PKMWide.query_normalization"):
                if self.query_norm_layer is not None:
                    x = self.query_norm_layer(x)

            B, S, Q = x.shape
            H = self.heads
            per_head = self.query_dim // H
            V = self.value_dim

        # Reshape for multi-head batched PKM: [B, S, H, per_head]
        # Then flatten H with query_dim for the batched PKM: [B, S, H, F*D]
            x = x.view(B, S, H, per_head)

        # Single vectorized PKM call for all heads (no Python loop)
            if return_aux:
                val, aux_h = self.pkm(x, return_aux=True)
            else:
                val = self.pkm(x)
                aux_h = None

        # val: [B, S, H, V]
            if self.value_norm_layer is not None:
                val = self.value_norm_layer(val)

        # Merge heads: [B, S, H * value_dim]
            merged = val.reshape(B, S, H * V)

            with _profile_range("PKMWide.output_projection"):
                output = self.output_proj(merged)

        # Output normalization
            with _profile_range("PKMWide.output_normalization"):
                if self.output_norm_layer is not None:
                    output = self.output_norm_layer(output)

        # Gating / residual integration
            with _profile_range("PKMWide.output_integration"):
                if self.output_mode == "residual":
                    output = original + output
                elif self.output_mode == "gated":
                    gate = torch.sigmoid(self.gate)
                    output = original + gate * output
        # output_mode == "none": return output as-is

        if return_branches:
            # Stack per-head outputs for regularization
            stacked = val.permute(2, 0, 1, 3)  # [H, B, S, V]
            if return_aux:
                aux_data = {"per_head_aux": [aux_h] if aux_h else [], "per_head_outputs": [val] if val is not None else []}
                return output, stacked, aux_data
            return output, stacked

        if return_aux:
            aux_data = {"per_head_aux": [aux_h], "per_head_outputs": [val]}
            return output, aux_data

        return output

    def forward_with_branches(self, x: torch.Tensor):
        """Return the merged output and branch (per-head) outputs."""
        result = self.forward(x, return_branches=True, return_aux=False)
        # Returns (output, stacked_branches)
        return result

    def forward_with_aux(self, x: torch.Tensor):
        """Return the merged output along with retrieval diagnostics."""
        return self.forward(x, return_branches=False, return_aux=True)

    def estimate_flops(self, batch: int = 1, seq: int = 1) -> int | None:
        total = 0
        encoder_flops = self.query_encoder.estimate_flops(batch, seq)
        total += encoder_flops
        total += self.pkm.estimate_flops(batch, seq) or 0
        # Output projection: 2 * batch * seq * (heads * value_dim) * dim
        total += 2 * batch * seq * self.heads * self.value_dim * self.dim
        return total

    def knowledge_parameters(self) -> Iterable:
        """Yield all knowledge-path parameters for training."""
        yield from self.parameters()

    def reasoning_parameters(self) -> Iterable:
        """PKMWide has no separate reasoning path; returns nothing."""
        return
        yield  # make generator

    def freeze_knowledge(self) -> "PKMWide":
        for p in self.knowledge_parameters():
            p.requires_grad_(False)
        return self

    def freeze_reasoning(self) -> "PKMWide":
        for p in self.reasoning_parameters():
            p.requires_grad_(False)
        return self

    def unfreeze_knowledge(self) -> "PKMWide":
        for p in self.knowledge_parameters():
            p.requires_grad_(True)
        return self

    def unfreeze_reasoning(self) -> "PKMWide":
        for p in self.reasoning_parameters():
            p.requires_grad_(True)
        return self

    def all_pkm_statistics(self) -> dict[str, Any]:
        """Collect key/duplicate statistics from the single batched PKM."""
        stats = {"heads": {}}
        pkm_stats = self.pkm.key_statistics()
        dup_stats = self.pkm.duplicate_statistics()
        stats["heads"][0] = {
            "key_stats": pkm_stats,
            "duplicates": dup_stats,
        }
        stats["total_duplicates"] = dup_stats["total_duplicates"]
        return stats

    def parameter_count(self) -> dict[str, int]:
        """Return parameter counts by component."""
        counts = {
            "total": 0,
            "query_encoder": 0,
            "pkm_keys": 0,
            "pkm_values": 0,
            "output_proj": 0,
            "norms": 0,
            "gate": 0,
        }
        for name, p in self.named_parameters():
            n = p.numel()
            counts["total"] += n
            if "query_encoder" in name:
                counts["query_encoder"] += n
            if "keys" in name:
                counts["pkm_keys"] += n
            if "values" in name:
                counts["pkm_values"] += n
            if "output_proj" in name:
                counts["output_proj"] += n
            if "norm" in name.lower():
                counts["norms"] += n
            if name == "gate":
                counts["gate"] += n
        return counts

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, memory_size={self.memory_size}, "
            f"query_dim={self.query_dim}, value_dim={self.value_dim}, "
            f"num_factors={self.num_factors}, heads={self.heads}, "
            f"topk={self.topk}, similarity={self.similarity!r}, "
            f"output_mode={self.output_mode!r}"
        )
