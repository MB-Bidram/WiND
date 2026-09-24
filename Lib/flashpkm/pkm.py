"""Modern Vectorized Product Key Memory with efficient top-k retrieval.

PKM uses factorized sub-key tables to enable efficient top-k retrieval over
a large effective memory without materializing the full memory in attention.

This implementation is fully vectorized: multi-head retrieval is done in a
single batched einsum + topk + gather call, with no Python loops over heads.

Architecture (single forward pass):
    1. Query [B, S, H, F*head_dim] is reshaped to [B, S, H, F, head_dim]
    2. Batched einsum with keys [H, F, M, head_dim] -> scores [B, S, H, F, M]
    3. Per-factor top-k: [B, S, H, F, k_f]
    4. Cartesian product via precomputed combos -> combo scores [B, S, H, n_combos]
    5. Global top-k: [B, S, H, G] selected combinations
    6. Batched value retrieval: values [H, M, F, V] gathered by selected memory
    7. Softmax-weighted aggregation of retrieved values -> output [B, S, H, V]
"""

from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from .kernels import (
    available as kernels_available,
    fused_value_aggregate_f2,
    hierarchical_topk,
)
from winc.modules import WindModule


def _profile_range(name: str):
    if torch.compiler.is_compiling():
        return nullcontext()
    if torch.autograd._profiler_enabled():
        return torch.profiler.record_function(name)
    return nullcontext()


class FactorizedPKM(WindModule):
    """Modern Vectorized Factorized Product Key Memory with multi-head support.

    Processes all ``heads`` in a single batched tensor operation.
    Keys and values are stored as ``[heads, num_factors, memory_size, dim]``
    and ``[heads, memory_size, num_factors, value_dim]`` respectively, enabling
    a single vectorized ``einsum`` instead of a Python loop over heads.

    Parameters
    ----------
    query_dim : int
        Dimensionality of each query token. Must equal head_dim * num_factors * heads.
    memory_size : int
        Number of slots per sub-key table.
    value_dim : int
        Dimensionality of each value vector.
    num_factors : int
        Number of factorized sub-key tables.
    head_dim : int or None
        Dimension of each sub-query.
    heads : int
        Number of retrieval heads (independent factorized PKM tables).
    topk : int
        Number of final candidates to retrieve globally per head.
    topk_per_factor : int or None
        Per-factor top-k before Cartesian-product scoring.
    similarity : str
        "dot" or "cosine" scoring.
    key_init_norm : bool
        Normalize sub-key rows to unit L2 norm at init.
    scale_factor : float or None
        Score scale. Defaults to 1/sqrt(subkey_dim).
    key_dtype : torch.dtype
        Storage dtype for quantized keys (int8/fp16/bf16/f32).
    value_dtype : torch.dtype
        Storage dtype for values.
    effective_memory_size : int or None
        Target Cartesian product-key address capacity. When supplied, the
        per-factor memory_size is derived as its ceiling nth root. This does
        not create an independent value vector for every Cartesian address;
        values retain the factor-additive PKM parameterization.
    """

    def __init__(
        self,
        query_dim: int,
        memory_size: int,
        value_dim: int,
        num_factors: int = 2,
        topk: int = 32,
        topk_per_factor: int | None = None,
        head_dim: int | None = None,
        subkey_dim: int | None = None,
        similarity: str = "cosine",
        heads: int = 1,
        scale_factor: float | None = None,
        key_init_norm: bool = True,
        key_dtype: torch.dtype = torch.float32,
        value_dtype: torch.dtype = torch.float32,
        effective_memory_size: int | None = None,
        exact_candidate_pruning: bool = False,
    ):
        super().__init__()
        if query_dim < 1:
            raise ValueError("query_dim must be positive")
        if effective_memory_size is not None:
            if effective_memory_size < 1:
                raise ValueError("effective_memory_size must be positive")
            memory_size = self._per_factor_memory_size(
                effective_memory_size, num_factors
            )
        if memory_size < 1:
            raise ValueError("memory_size must be positive")
        if value_dim < 1:
            raise ValueError("value_dim must be positive")
        if num_factors < 1:
            raise ValueError("num_factors must be at least 1")
        if heads < 1:
            raise ValueError("heads must be at least 1")
        if topk < 1:
            raise ValueError("topk must be positive")
        if similarity not in {"dot", "cosine"}:
            raise ValueError(f"unknown similarity: {similarity!r}")

        head_dim = head_dim or query_dim // (num_factors * heads)
        if query_dim != head_dim * num_factors * heads:
            raise ValueError(
                f"query_dim ({query_dim}) must equal head_dim*num_factors*heads "
                f"({head_dim}*{num_factors}*{heads}={head_dim * num_factors * heads})"
            )
        subkey_dim = subkey_dim or head_dim
        if subkey_dim != head_dim:
            raise ValueError("subkey_dim must equal head_dim")

        topk_per_factor = topk_per_factor or min(32, memory_size)
        if topk_per_factor > memory_size:
            topk_per_factor = memory_size

        n_combos = topk_per_factor ** num_factors
        _COMBO_LIMIT = 4096
        if n_combos > _COMBO_LIMIT:
            raise ValueError(
                f"topk_per_factor ({topk_per_factor}) ** num_factors ({num_factors}) "
                f"= {n_combos} exceeds the safe Cartesian-product limit "
                f"({_COMBO_LIMIT}). This would materialize a tensor of "
                f"[batch, seq_len, heads, {n_combos}, num_factors] per forward pass, "
                f"risking OOM. Reduce topk_per_factor or num_factors."
            )
        if topk > n_combos:
            raise ValueError(
                f"topk ({topk}) cannot exceed available Cartesian candidates ({n_combos})"
            )

        self.query_dim = query_dim
        self.memory_size = memory_size
        self.effective_memory_size = memory_size ** num_factors
        self.requested_effective_memory_size = effective_memory_size
        self.value_dim = value_dim
        self.num_factors = num_factors
        self.heads = heads
        self.topk = topk
        self.topk_per_factor = topk_per_factor
        self.head_dim = head_dim
        self.subkey_dim = subkey_dim
        self.similarity = similarity
        self.key_init_norm = key_init_norm
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        self.retrieval_backend = "auto"
        self.factor_topk_backend = "auto"
        self.exact_candidate_pruning = bool(exact_candidate_pruning)

        # Memory-efficient execution configuration
        self.efficient_mode = False  # Use tiled scoring and checkpointed retrieval
        self.key_tile_size = 512  # Keys per tile for streamed top-k
        self.query_tile_size = 64  # Queries per tile for checkpointed retrieval


        if scale_factor is None:
            scale_factor = 1.0 / math.sqrt(subkey_dim)
        self.scale_factor = scale_factor

        # Sub-key tables: [heads, num_factors, memory_size, subkey_dim]
        # Keys are always stored as float32 parameters for training.
        # key_dtype controls how they are processed in forward (quantization).
        self.keys = nn.Parameter(torch.empty(heads, num_factors, memory_size, subkey_dim))
        # Value memory: [heads, memory_size, num_factors, value_dim]
        self.values = nn.Parameter(torch.empty(heads, memory_size, num_factors, value_dim, dtype=value_dtype))

        self._init_weights()

        # Precompute the combos lookup table at init time since all parameters
        # are static. This avoids creating dynamic-shape tensors during forward
        # that confuse torch.compile's inductor backend.
        _total = topk_per_factor ** num_factors
        _combos = torch.zeros(_total, num_factors, dtype=torch.long)
        for _i in range(num_factors):
            _combos[:, _i] = (torch.arange(_total) // (topk_per_factor ** _i)) % topk_per_factor
        self.register_buffer("_combos", _combos, persistent=False)
        # For additive factor scores, a final top-G combination can never
        # contain an entry ranked below G in any individual factor: pairing
        # each of the G better entries with the best remaining factors already
        # dominates it.  Keep an optional smaller Cartesian table for that
        # exact candidate reduction.  It is opt-in and auxiliary diagnostics
        # retain the configured per-factor top-k contract.
        self._pruned_topk_per_factor = min(topk_per_factor, topk)
        _pruned_total = self._pruned_topk_per_factor ** num_factors
        _pruned_combos = torch.zeros(_pruned_total, num_factors, dtype=torch.long)
        for _i in range(num_factors):
            _pruned_combos[:, _i] = (
                torch.arange(_pruned_total) // (self._pruned_topk_per_factor ** _i)
            ) % self._pruned_topk_per_factor
        self.register_buffer("_pruned_combos", _pruned_combos, persistent=False)
        self.register_buffer(
            "_factor_offsets", torch.arange(num_factors, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_head_offsets",
            torch.arange(heads, dtype=torch.long) * memory_size * num_factors,
            persistent=False,
        )
        self.register_buffer("_head_indices", torch.arange(heads, dtype=torch.long), persistent=False)
        self._cached_normalized_keys: torch.Tensor | None = None
        self._cached_keys_version = -1
        self._cached_effective_keys: torch.Tensor | None = None
        self._cached_effective_keys_version = -1

        # Precompute quantization scale for int8 to avoid dynamic .max() in forward.
        # The scale is computed from the initial key distribution and stays fixed.
        if key_dtype == torch.int8:
            with torch.no_grad():
                _max_val = self.keys.abs().max().clamp_min(1e-6)
                _scale = _max_val / 127.0
            self.register_buffer("_quant_scale", _scale, persistent=False)
        else:
            # Register a dummy scale so the attribute always exists.
            self.register_buffer("_quant_scale", torch.tensor(1.0), persistent=False)

    @staticmethod
    def _per_factor_memory_size(
        effective_memory_size: int, num_factors: int
    ) -> int:
        memory_size = max(1, int(effective_memory_size ** (1.0 / num_factors)))
        while memory_size ** num_factors < effective_memory_size:
            memory_size += 1
        while (
            memory_size > 1
            and (memory_size - 1) ** num_factors >= effective_memory_size
        ):
            memory_size -= 1
        return memory_size

    def _apply(self, fn):
        self._invalidate_inference_key_cache()
        return super()._apply(fn)

    def _invalidate_inference_key_cache(self) -> None:
        """Drop derived inference keys after a state or mode transition.

        Optimizers are allowed to update parameters through paths that do not
        reliably advance ``Parameter._version``.  Entering training is an
        unambiguous invalidation boundary, so this protects the next eval pass
        from reusing normalized/scaled keys created before an optimizer step.
        """
        self._cached_normalized_keys = None
        self._cached_keys_version = -1
        self._cached_effective_keys = None
        self._cached_effective_keys_version = -1

    def train(self, mode: bool = True):
        if mode:
            self._invalidate_inference_key_cache()
        return super().train(mode)

    def _inference_effective_keys(self, *, normalize: bool) -> torch.Tensor:
        """Return cached inference keys with normalization and score scale fused.

        Moving ``scale_factor`` to the static key table removes the per-forward
        query multiply while preserving the same dot-product computation.
        The parameter version invalidates the cache after an optimizer update or
        an in-place checkpoint load.
        """
        if torch.compiler.is_compiling() or self._has_execution_adapter("keys"):
            if self._cached_effective_keys is not None:
                return self._cached_effective_keys
            keys = self._dequant_keys()
            if normalize:
                keys = F.normalize(keys, dim=-1, p=2.0)
            return keys * self.scale_factor

        keys_version = self.keys._version
        if (
            self._cached_effective_keys is None
            or self._cached_effective_keys_version != keys_version
            or self._cached_effective_keys.device != self.keys.device
        ):
            keys = self._dequant_keys()
            if normalize:
                keys = F.normalize(keys, dim=-1, p=2.0)
            self._cached_effective_keys = keys * self.scale_factor
            self._cached_effective_keys_version = keys_version
        return self._cached_effective_keys

    def set_retrieval_backend(self, backend: str = "auto") -> "FactorizedPKM":
        """Choose exact inference retrieval: ``auto``, ``torch``, or ``triton``."""
        if backend not in {"auto", "torch", "triton"}:
            raise ValueError(f"unknown retrieval backend: {backend!r}")
        self.retrieval_backend = backend
        return self

    def set_factor_topk_backend(self, backend: str = "auto") -> "FactorizedPKM":
        """Choose per-factor exact top-k: ``auto``, ``torch``, or ``hierarchical``.

        ``auto`` currently dispatches to PyTorch/CUB because its exact top-k
        implementation outperforms the hierarchical kernel on all benchmarked
        production shapes. ``hierarchical`` remains explicit for research and
        targeted hardware experiments.
        """
        if backend not in {"auto", "torch", "hierarchical"}:
            raise ValueError(f"unknown factor top-k backend: {backend!r}")
        self.factor_topk_backend = backend
        return self

    def prepare_for_inference(
        self,
        dtype: torch.dtype | None = None,
        retrieval_backend: str = "auto",
        factor_topk_backend: str = "auto",
    ) -> "FactorizedPKM":
        super().prepare_for_inference(dtype)
        with torch.inference_mode():
            self._inference_effective_keys(normalize=self.similarity == "cosine")
        return self.set_retrieval_backend(retrieval_backend).set_factor_topk_backend(
            factor_topk_backend
        )

    def set_efficient_mode(
        self, 
        enabled: bool = True,
        key_tile_size: int = 512,
        query_tile_size: int = 64,
    ) -> "FactorizedPKM":
        """Enable memory-efficient tiled scoring and checkpointed retrieval.
        
        Args:
            enabled: Whether to use efficient mode
            key_tile_size: Keys processed per tile in streaming top-k
            query_tile_size: Queries per checkpoint tile
        """
        self.efficient_mode = enabled
        self.key_tile_size = max(1, key_tile_size)
        self.query_tile_size = max(1, query_tile_size)
        return self

    @torch.inference_mode()
    def infer(self, query: torch.Tensor) -> torch.Tensor:
        """Run the configured no-grad inference path."""
        return self(query)

    def _dequant_keys(self) -> torch.Tensor:
        """Return keys in computation dtype, applying quantization if configured.

        Note: When key_dtype is torch.int8, this performs simulated
        quantization (quantize-dequantize) for training purposes. The keys
        parameter itself remains float32 for optimizer compatibility. This
        provides quantization-aware training simulation, NOT actual int8
        memory savings. True int8 storage would require a custom optimizer
        or dequantization wrapper that updates float32 master weights.
        """
        keys = self._execution_parameter("keys", self.keys)
        if self.key_dtype == torch.int8:
            # Use a fixed scale to avoid dynamic .max() that confuses inductor.
            # The scale is computed once at init and registered as a buffer.
            scale = self._quant_scale
            quantized = keys / scale
            rounded = torch.round(quantized).clamp(-128, 127)
            if self.training:
                # Straight-through estimator: dequantized forward, identity derivative
                return (quantized + (rounded - quantized).detach()) * scale
            return rounded.to(torch.float32) * scale
        elif self.key_dtype == torch.float16:
            return keys.half()
        elif self.key_dtype == torch.bfloat16:
            return keys.bfloat16()
        return keys

    def _has_execution_adapter(self, name: str) -> bool:
        adapters = getattr(self, "_wind_lora_parameter_adapters", None)
        return adapters is not None and f"p_{name}" in adapters

    def _execution_parameter(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Apply an optional architecture-owned adapter without changing base state."""
        adapters = getattr(self, "_wind_lora_parameter_adapters", None)
        if adapters is None or f"p_{name}" not in adapters:
            return value
        return adapters[f"p_{name}"](value)

    def _init_weights(self) -> None:
        nn.init.normal_(self.keys)
        if self.key_init_norm:
            with torch.no_grad():
                norms = self.keys.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                self.keys.div_(norms)
        nn.init.normal_(self.values, std=0.01)

    def _compute_factor_scores_dense(
        self, query: torch.Tensor, *, return_scores: bool = False,
        topk_per_factor: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Compute per-factor top-k scores using full materialized scoring.
        
        This is the original dense implementation retained for diagnostics,
        dense mode, and as a correctness reference.

        Args:
            query: [B, S, H, F*D] query tensor (H must match self.heads),
                   or [B, S, F*D] for single-head (H=1) backward compatibility.
            return_scores: If True, also return full per-factor scores (costly).

        Returns:
            topk_scores:  [B, S, H, F, k_f]
            topk_indices: [B, S, H, F, k_f] (long)
            sub_scores:   [B, S, H, F, M] if return_scores else None
        """
        if query.dim() == 3:
            B, S, FD = query.shape
            H = 1
            query = query.unsqueeze(2)  # [B, S, 1, FD]
        else:
            B, S, H, FD = query.shape

        if H != self.heads:
            raise ValueError(f"expected {self.heads} heads, got {H}")
        expected_dim = self.num_factors * self.head_dim
        if FD != expected_dim:
            raise ValueError(f"expected final query dimension {expected_dim}, got {FD}")

        n_factors = self.num_factors
        factor_topk = topk_per_factor or self.topk_per_factor
        d = self.head_dim

        q = query.reshape(B, S, H, n_factors, d)

        inference = not self.training and not torch.is_grad_enabled()
        if self.similarity == "cosine":
            q = F.normalize(q, dim=-1, p=2.0)
            if inference:
                keys = self._inference_effective_keys(normalize=True)
            else:
                keys = F.normalize(self._dequant_keys(), dim=-1, p=2.0)
                q = q * self.scale_factor
        else:
            if inference:
                keys = self._inference_effective_keys(normalize=False)
            else:
                keys = self._dequant_keys()
                q = q * self.scale_factor

        if keys.dtype != q.dtype:
            keys = keys.to(q.dtype)

        # Keep factors and heads as the batch dimension for cuBLAS.  This is
        # equivalent to the einsum below, but avoids the generic einsum
        # contraction path and is consistently faster for the supported PKM
        # score layouts:
        # [B, S, H, F, D] @ [H, F, M, D] -> [B, S, H, F, M].
        with _profile_range("PKM.factor_score"):
            q_bmm = q.permute(2, 3, 0, 1, 4).reshape(H * n_factors, B * S, d)
            keys_bmm = keys.reshape(
                H * n_factors, self.memory_size, d
            ).transpose(1, 2)
            score_rows = torch.bmm(q_bmm, keys_bmm)
        use_hierarchical_topk = self.factor_topk_backend == "hierarchical"
        topk_range = (
            "PKM.factor_topk.batched_f2"
            if n_factors == 2
            else "PKM.factor_topk.batched"
        )
        with _profile_range(topk_range):
            if use_hierarchical_topk:
                if torch.is_grad_enabled():
                    raise RuntimeError("hierarchical factor top-k is inference-only")
                scores = score_rows.reshape(
                    H, n_factors, B, S, self.memory_size
                ).permute(2, 3, 0, 1, 4)
                topk_scores, topk_indices = hierarchical_topk(scores, factor_topk)
            else:
                topk_scores, topk_indices = torch.topk(
                    score_rows, k=factor_topk, dim=-1
                )
                topk_scores = topk_scores.reshape(
                    H, n_factors, B, S, factor_topk
                ).permute(2, 3, 0, 1, 4)
                topk_indices = topk_indices.reshape(
                    H, n_factors, B, S, factor_topk
                ).permute(2, 3, 0, 1, 4)

        sub_scores = (
            score_rows.reshape(H, n_factors, B, S, self.memory_size).permute(
                2, 3, 0, 1, 4
            )
            if return_scores
            else None
        )
        return topk_scores, topk_indices, sub_scores

    def _compute_factor_scores_streamed(
        self,
        query: torch.Tensor,
        *,
        topk_per_factor: int | None = None,
        recompute_selected: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-factor top-k using tiled streaming to bound memory.
        
        Processes keys in tiles, maintains running top-k, then recomputes
        selected scores with autograd for gradient flow.
        
        Args:
            query: [B, S, H, F, D] normalized and scaled query
            topk_per_factor: Per-factor k (defaults to self.topk_per_factor)
            recompute_selected: Recompute selected scores with grad
            
        Returns:
            topk_scores: [B, S, H, F, k_f]
            topk_indices: [B, S, H, F, k_f]
        """
        B, S, H, n_factors, d = query.shape
        M = self.memory_size
        factor_topk = topk_per_factor or self.topk_per_factor
        tile_size = min(self.key_tile_size, M)
        
        # Get keys (already preprocessed for similarity mode by caller)
        inference = not self.training and not torch.is_grad_enabled()
        if inference:
            keys = self._inference_effective_keys(normalize=self.similarity == "cosine")
        else:
            keys = self._dequant_keys()
            if self.similarity == "cosine":
                keys = F.normalize(keys, dim=-1, p=2.0)
        
        if keys.dtype != query.dtype:
            keys = keys.to(query.dtype)
        
        # Reshape for batched processing
        q_bmm = query.permute(2, 3, 0, 1, 4).reshape(H * n_factors, B * S, d)
        
        # Stream through key tiles, maintain running top-k
        num_tiles = (M + tile_size - 1) // tile_size
        running_scores = None
        running_indices = None
        
        with torch.no_grad() if recompute_selected else nullcontext():
            for tile_idx in range(num_tiles):
                start_idx = tile_idx * tile_size
                end_idx = min(start_idx + tile_size, M)
                tile_keys = keys[:, :, start_idx:end_idx, :].reshape(
                    H * n_factors, end_idx - start_idx, d
                ).transpose(1, 2)
                
                # Score this tile
                tile_scores = torch.bmm(q_bmm, tile_keys)  # [H*F, B*S, tile_size]
                
                # Get top-k from this tile
                k_tile = min(factor_topk, tile_scores.shape[-1])
                tile_topk_scores, tile_topk_indices = torch.topk(
                    tile_scores, k=k_tile, dim=-1
                )
                # Adjust indices to global key space
                tile_topk_indices = tile_topk_indices + start_idx
                
                # Merge with running top-k
                if running_scores is None:
                    running_scores = tile_topk_scores
                    running_indices = tile_topk_indices
                else:
                    # Concatenate and re-select top-k
                    combined_scores = torch.cat([running_scores, tile_topk_scores], dim=-1)
                    combined_indices = torch.cat([running_indices, tile_topk_indices], dim=-1)
                    k_combined = min(factor_topk, combined_scores.shape[-1])
                    running_scores, top_positions = torch.topk(
                        combined_scores, k=k_combined, dim=-1
                    )
                    running_indices = torch.gather(combined_indices, -1, top_positions)
        
        # Reshape to [B, S, H, F, k_f]
        topk_indices = running_indices.reshape(H, n_factors, B, S, factor_topk).permute(
            2, 3, 0, 1, 4
        )
        
        if not recompute_selected:
            topk_scores = running_scores.reshape(H, n_factors, B, S, factor_topk).permute(
                2, 3, 0, 1, 4
            )
            return topk_scores, topk_indices
        
        # Recompute selected scores with autograd enabled
        # Use gather on full keys (one gather per factor is cheaper than tiling again)
        with _profile_range("PKM.recompute_selected_scores"):
            keys_bmm = keys.reshape(H * n_factors, M, d).transpose(1, 2)
            # q_bmm: [H*F, B*S, d]
            # We need to gather specific keys and compute dot products
            
            # Reshape indices for gathering: [H*F, B*S, k_f]
            gather_indices = topk_indices.permute(2, 3, 0, 1, 4).reshape(
                H * n_factors, B * S, factor_topk
            )
            
            # Gather keys: need to expand for broadcasting
            # keys_bmm: [H*F, d, M] -> [H*F, M, d] for gathering
            keys_for_gather = keys_bmm.transpose(1, 2)  # [H*F, M, d]
            
            # Expand indices for gathering along last dim
            gather_indices_expanded = gather_indices.unsqueeze(-1).expand(
                H * n_factors, B * S, factor_topk, d
            )
            
            # Gather: [H*F, B*S, k_f, d]
            selected_keys = torch.gather(
                keys_for_gather.unsqueeze(1).expand(H * n_factors, B * S, M, d),
                2,
                gather_indices_expanded
            )
            
            # Compute scores: [H*F, B*S, k_f]
            recomputed_scores = (q_bmm.unsqueeze(-1) * selected_keys.transpose(2, 3)).sum(dim=2)
            
            topk_scores = recomputed_scores.reshape(
                H, n_factors, B, S, factor_topk
            ).permute(2, 3, 0, 1, 4)
        
        return topk_scores, topk_indices

    def forward(
        self,
        query: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Retrieve from memory and aggregate values.

        Args:
            query: [B, S, H, F*D] query tensor (H must match self.heads),
                or [B, S, F*D] for single-head (H=1) backward compatibility.
            return_aux: Return full diagnostic tensors (forces dense mode).
        
        Dispatches to:
            - Dense mode: return_aux=True, or efficient_mode=False
            - Efficient mode: efficient_mode=True and return_aux=False
        """
        # Full diagnostics always use dense path (they return large tensors anyway)
        if return_aux or not self.efficient_mode:
            return self._forward_dense(query, return_aux=return_aux)
        
        # Efficient tiled+checkpointed path
        return self._forward_efficient(query)

    def _forward_dense(
        self,
        query: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Original dense retrieval implementation.
        
        Retained as the reference implementation and for full diagnostics.
        """
        # Handle both 3D [B, S, FD] (single head) and 4D [B, S, H, FD] (multi-head)
        if query.dim() == 3:
            B, S, FD = query.shape
            H = 1
            query = query.unsqueeze(2)  # [B, S, 1, FD]
            _was_3d = True
        else:
            B, S, H, FD = query.shape
            _was_3d = False

        n_factors = self.num_factors
        M = self.memory_size
        V = self.value_dim
        G = self.topk
        k_f = self.topk_per_factor
        candidate_k_f = (
            self._pruned_topk_per_factor
            if self.exact_candidate_pruning and not return_aux
            else k_f
        )

        with _profile_range("FlashPKM.forward.dense"):
            topk_scores, topk_indices, sub_scores = self._compute_factor_scores_dense(
                query, return_scores=return_aux, topk_per_factor=candidate_k_f
            )

            # [Rest of the original forward body - copy everything from the current
            # forward() starting from the "# Global top-k from Cartesian product" comment
            # through to the final return statement]
            
            # Global top-k from Cartesian product (batched over H)
            combos = (
                self._pruned_combos if candidate_k_f == self._pruned_topk_per_factor
                else self._combos
            )
            n_combos = combos.shape[0]

            ts = topk_scores.reshape(B * S * H, n_factors, candidate_k_f)
            ti = topk_indices.reshape(B * S * H, n_factors, candidate_k_f)
            n_bsh = B * S * H

            with _profile_range("PKM.candidate_generation"):
                if n_factors == 2:
                    pair_scores = ts[:, 0, :, None] + ts[:, 1, None, :]
                else:
                    batch_indices = torch.arange(n_bsh, device=query.device)[:, None, None]
                    factor_indices = torch.arange(n_factors, device=query.device)[None, None, :]
                    pair_scores = ts[batch_indices, factor_indices, combos].sum(dim=-1)

            with _profile_range("PKM.candidate_topk"):
                selected_scores_flat, selected_combos_flat = torch.topk(
                    pair_scores.flatten(1), k=G, dim=-1
                )
            with _profile_range("PKM.value_gather"):
                if n_factors == 2:
                    selected_factors = torch.stack(
                        (
                            selected_combos_flat.div(candidate_k_f, rounding_mode="floor"),
                            selected_combos_flat.remainder(candidate_k_f),
                        ),
                        dim=-1,
                    )
                else:
                    selected_factors = combos[selected_combos_flat]
                tf = selected_factors.reshape(n_bsh, G, n_factors)
                selected_memory_flat = torch.gather(ti.transpose(1, 2), 1, tf)
                selected_memory = selected_memory_flat.reshape(B, S, H, G, n_factors)

            selected_scores = selected_scores_flat.reshape(B, S, H, G)

            adapters = getattr(self, "_wind_lora_parameter_adapters", None)
            value_adapter = None if adapters is None or "p_values" not in adapters else adapters["p_values"]
            selected_value_adapter = (
                value_adapter is not None
                and hasattr(value_adapter, "gather_values")
                and not value_adapter.merged
                and value_adapter.enabled
            )
            values = self.values if selected_value_adapter else self._execution_parameter("values", self.values)
            can_fuse_retrieval = (
                n_factors == 2
                and not return_aux
                and not self.training
                and not torch.is_grad_enabled()
                and kernels_available()
                and query.device.type == "cuda"
                and values.device.type == "cuda"
                and not selected_value_adapter
            )
            if self.retrieval_backend == "triton" and not can_fuse_retrieval:
                raise RuntimeError(
                    "Triton retrieval requires CUDA, eval mode, no-grad execution, "
                    "return_aux=False, and num_factors=2"
                )
            if can_fuse_retrieval and self.retrieval_backend != "torch":
                with _profile_range("PKM.value_aggregation.triton_f2"):
                    output = fused_value_aggregate_f2(
                        selected_scores_flat,
                        selected_memory_flat,
                        values,
                    ).reshape(B, S, H, V)
                if _was_3d:
                    output = output.squeeze(2)
                return output

            with _profile_range("PKM.value_lookup"):
                if n_factors == 2:
                    head_indices = self._head_indices.view(1, 1, H).expand(B, S, H).reshape(n_bsh)
                    retrieved_values = (
                        values[head_indices[:, None], selected_memory_flat[:, :, 0], 0]
                        + values[head_indices[:, None], selected_memory_flat[:, :, 1], 1]
                    ).reshape(B, S, H, G, V)
                    if selected_value_adapter:
                        delta = value_adapter.gather_values(head_indices, selected_memory_flat)
                        retrieved_values = retrieved_values + delta.reshape(B, S, H, G, V).to(retrieved_values.dtype)
                else:
                    values_flat_1d = values.reshape(H * M * n_factors, V)
                    combined_idx = (
                        selected_memory * n_factors
                        + self._factor_offsets.view(1, 1, 1, 1, n_factors)
                        + self._head_offsets.view(1, 1, H, 1, 1)
                    )
                    idx_flat = combined_idx.reshape(n_bsh, G * n_factors)
                    retrieved_values = values_flat_1d[idx_flat].reshape(n_bsh, G, n_factors, V).sum(dim=2)
                    retrieved_values = retrieved_values.reshape(B, S, H, G, V)
                    if selected_value_adapter:
                        head_indices = self._head_indices.view(1, 1, H).expand(B, S, H).reshape(n_bsh)
                        delta = value_adapter.gather_values(head_indices, selected_memory_flat)
                        retrieved_values = retrieved_values + delta.reshape(B, S, H, G, V).to(retrieved_values.dtype)

            if retrieved_values.dtype != query.dtype:
                retrieved_values = retrieved_values.to(query.dtype)

            with _profile_range("PKM.value_aggregation.aten"):
                weights = F.softmax(selected_scores, dim=-1)
                output = torch.matmul(weights.unsqueeze(-2), retrieved_values).squeeze(-2)

            if _was_3d:
                output = output.squeeze(2)

            if return_aux:
                aux = {
                    "topk_scores": selected_scores,
                    "topk_indices": selected_memory,
                    "sub_scores": sub_scores,
                    "retrieval_weights": weights,
                    "retrieved_values": retrieved_values,
                    "topk_per_factor": self.topk_per_factor,
                    "num_factors": self.num_factors,
                    "heads": self.heads,
                }
                return output, aux
            return output


    def estimate_flops(self, batch: int = 1, seq: int = 1) -> int | None:
        n_factors = self.num_factors
        H = self.heads
        M = self.memory_size
        k = self.topk_per_factor
        n_combos = k ** n_factors
        G = self.topk
        scores = batch * seq * H * n_factors * M * self.subkey_dim * 2
        combos = batch * seq * H * n_combos * n_factors * 2
        retrieval = batch * seq * H * G * self.value_dim * 2
        return scores + combos + retrieval

    @torch.no_grad()
    def key_statistics(self) -> dict[str, Any]:
        """Return per-factor key statistics for diagnostics."""
        keys = self.keys
        norms = keys.norm(dim=-1)
        results: dict[str, Any] = {"factors": {}, "heads": self.heads}

        for f in range(self.num_factors):
            # Aggregate stats across all heads
            fkeys_h0 = keys[0, f]
            f_norms_h0 = norms[0, f]
            if self.similarity == "cosine":
                fkeys_n = F.normalize(fkeys_h0, dim=-1, p=2.0)
            else:
                fkeys_n = fkeys_h0
            sim = torch.matmul(fkeys_n, fkeys_n.transpose(-2, -1))
            eye = torch.eye(self.memory_size, device=self.keys.device, dtype=torch.bool)
            sim = sim[~eye].abs()
            results["factors"][f"factor_{f}"] = {
                "min_norm": float(f_norms_h0.min()),
                "max_norm": float(f_norms_h0.max()),
                "mean_norm": float(f_norms_h0.mean()),
                "max_pairwise_sim": float(sim.max()) if sim.numel() > 0 else 0.0,
                "mean_pairwise_sim": float(sim.mean()) if sim.numel() > 0 else 0.0,
            }

        results["memory_size"] = self.memory_size
        results["num_factors"] = self.num_factors
        results["heads"] = self.heads
        results["key_dim"] = self.subkey_dim
        return results

    @torch.no_grad()
    def duplicate_statistics(self, threshold: float = 0.99) -> dict[str, Any]:
        """Count near-duplicate sub-keys per factor.

        Note: This computes a full pairwise similarity matrix per factor
        (O(memory_size²)), making it unsuitable for hot paths. It is intended
        for offline diagnostics only — never call during training or inference.
        """
        results: dict[str, Any] = {"threshold": threshold, "factors": {}, "total_duplicates": 0}
        total = 0

        for f in range(self.num_factors):
            # Check per head, aggregate counts
            dup_count_total = 0
            dup_slots_all = []
            for h in range(self.heads):
                fkeys = self.keys[h, f]
                fkeys_n = F.normalize(fkeys, dim=-1, p=2.0)
                sim = torch.matmul(fkeys_n, fkeys_n.transpose(-2, -1))
                eye = torch.eye(self.memory_size, device=self.keys.device, dtype=torch.bool)
                sim = sim.masked_fill(eye, 0.0)
                dup_mask = sim > threshold
                dup_count = dup_mask.sum().item() // 2
                dup_count_total += dup_count
                dup_rows = dup_mask.any(dim=-1).nonzero(as_tuple=False).squeeze(-1)
                if len(dup_rows) > 0:
                    dup_slots_all.append(dup_rows.tolist())

            total += dup_count_total
            results["factors"][f"factor_{f}"] = {
                "duplicate_pairs": dup_count_total,
                "duplicate_slots": dup_slots_all,
                "num_duplicate_slots": len(dup_slots_all),
            }

        results["total_duplicates"] = total
        return results

    @torch.no_grad()
    def find_duplicate_keys(self, threshold: float = 0.99) -> dict[int, list[torch.Tensor]]:
        """Find indices of duplicate sub-keys per factor.

        O(memory_size²) pairwise similarity — offline diagnostics only.
        """
        result: dict[int, list[torch.Tensor]] = {}
        for f in range(self.num_factors):
            head_dups = []
            for h in range(self.heads):
                fkeys = self.keys[h, f]
                fkeys_n = F.normalize(fkeys, dim=-1, p=2.0)
                sim = torch.matmul(fkeys_n, fkeys_n.transpose(-2, -1))
                eye = torch.eye(self.memory_size, device=self.keys.device, dtype=torch.bool)
                sim = sim.masked_fill(eye, 0.0)
                dup_rows = (sim > threshold).any(dim=-1).nonzero(as_tuple=False).squeeze(-1)
                head_dups.append((h, dup_rows))
            result[f] = head_dups
        return result

    @torch.no_grad()
    def deduplicate(self, threshold: float = 0.99, mode: str = "replace") -> int:
        """Explicitly remediate duplicate sub-keys.

        Warning: modifies key memory in-place. Call outside training loops.
        O(memory_size²) — offline maintenance only.
        """
        if mode not in {"replace", "mark", "merge"}:
            raise ValueError(f"unknown deduplication mode: {mode!r}")
        if mode == "mark":
            return sum(len(v) for v in self.find_duplicate_keys(threshold).values())
        modified = 0
        for f in range(self.num_factors):
            head_dups = self.find_duplicate_keys(threshold)[f]
            for h, dup_indices in head_dups:
                if len(dup_indices) == 0:
                    continue
                if mode == "replace":
                    dup_keys = torch.randn_like(self.keys[h, f][dup_indices])
                    if self.key_init_norm:
                        norms = dup_keys.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        dup_keys.div_(norms)
                    self.keys.data[h, f, dup_indices] = dup_keys
                elif mode == "merge":
                    new_keys = torch.randn_like(self.keys[h, f][dup_indices])
                    norms = new_keys.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    new_keys.div_(norms)
                    self.keys.data[h, f, dup_indices] = new_keys
                modified += len(dup_indices)
        return modified

    def extra_repr(self) -> str:
        return (
            f"query_dim={self.query_dim}, memory_size={self.memory_size}, "
            f"value_dim={self.value_dim}, num_factors={self.num_factors}, "
            f"heads={self.heads}, topk={self.topk}, "
            f"topk_per_factor={self.topk_per_factor}, "
            f"head_dim={self.head_dim}, subkey_dim={self.subkey_dim}, "
            f"similarity={self.similarity!r}"
        )
