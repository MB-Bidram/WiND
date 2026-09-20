"""Fused Triton kernels for PKM acceleration.

These kernels fuse factor-scoring and topk selection into single CUDA kernels
to maximize performance. All kernel constants must be tl.constexpr.

Requirements: torch with CUDA support and triton >= 2.1
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_SCORE_CONFIGS = [
    triton.Config({"BLOCK_M": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 256}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 256}, num_warps=8, num_stages=3),
]

_RETRIEVE_CONFIGS = [
    triton.Config({"BLOCK_V": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_V": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_V": 256}, num_warps=4, num_stages=3),
]

_HIERARCHICAL_TOPK_CONFIGS = [
    triton.Config({}, num_warps=2, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
]

@triton.autotune(configs=_SCORE_CONFIGS, key=["B", "S", "F", "D", "M"])
@triton.jit
def pkm_score_kernel(
    query_ptr,
    keys_ptr,
    scores_ptr,
    scale_val: tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Fused factor scoring kernel.

    For each (b, s, f), computes scores against all M memory slots.

    This fuses the einsum(query, keys) operation into a single kernel.
    """
    off_bsf = tl.program_id(0)
    off_m_block = tl.program_id(1)
    off_b = off_bsf // (S * F)
    off_s = (off_bsf // F) % S
    off_f = off_bsf % F

    q_base = query_ptr + off_b * S * F * D + off_s * F * D + off_f * D
    k_base = keys_ptr + off_f * M * D
    scores_base = scores_ptr + off_b * S * F * M + off_s * F * M + off_f * M

    # Load query [D]
    d_idx = tl.arange(0, BLOCK_D)
    q_mask = d_idx < D
    q = tl.load(q_base + d_idx, mask=q_mask, other=0.0)

    # Each program handles a bounded M tile. The previous implementation used
    # one program for every M entry, causing a 4096 x D live tile and severe
    # register spilling for common D=64, M=4096 workloads.
    m_idx = off_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_idx < M
    d_idx2 = tl.arange(0, BLOCK_D)
    k_ptrs = k_base + m_idx[:, None] * D + d_idx2[None, :]
    k_mask = m_mask[:, None] & (d_idx2[None, :] < D)
    keys = tl.load(k_ptrs, mask=k_mask, other=0.0)

    # Compute scores [BLOCK_M] = dot(query, keys[i])
    scores = tl.sum(q[None, :] * keys, axis=1) * scale_val

    # Store scores
    tl.store(scores_base + m_idx, scores, mask=m_mask)


@triton.jit
def pkm_retrieve_kernel(
    topk_indices_ptr,
    values_ptr,
    output_ptr,
    B: tl.constexpr,
    S: tl.constexpr,
    G: tl.constexpr,
    n_factors: tl.constexpr,
    M: tl.constexpr,
    V: tl.constexpr,
    k_f: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_KF: tl.constexpr,
):
    """Fused value retrieval from memory.

    WARNING: UNIMPLEMENTED PLACEHOLDER.

    This kernel is defined for interface completeness but is NOT wired into
    the PyTorch retrieval path. The current implementation relies on
    ``torch.gather`` which is already well-optimized. This kernel should
    only be used after benchmarking confirms it provides a measurable
    speedup over ``torch.gather`` for the specific access patterns in
    ``FactorizedPKM.forward``.

    Do not call this kernel — use ``fused_factor_scoring`` (which is active)
    or the pure-PyTorch path instead.
    """
    off_b = tl.program_id(0)
    off_s = tl.program_id(1)
    off_g = tl.program_id(2)

    if off_b >= B or off_s >= S or off_g >= G:
        return

    # Load topk indices for this position
    # topk_indices: [B, S, n_factors, k_f] -> need [B, S, G, n_factors] from _select_global_topk
    # Actually, we receive selected_memory: [B, S, G, n_factors]
    # This kernel assumes indices are pre-processed

    # Output pointer
    out_base = output_ptr + off_b * S * G * V + off_s * G * V + off_g * V

    # For now, this is a placeholder - the gather operation is complex
    # and better handled by torch.gather which is already optimized


@triton.autotune(configs=_RETRIEVE_CONFIGS, key=["G", "V"])
@triton.jit
def pkm_retrieve_aggregate_f2_kernel(
    selected_scores_ptr,
    selected_memory_ptr,
    values_ptr,
    output_ptr,
    H: tl.constexpr,
    M: tl.constexpr,
    G: tl.constexpr,
    V: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Fuse F=2 softmax, value lookup, and weighted aggregation.

    The kernel is inference-only: PyTorch retains the autograd-capable path.
    Each program owns one flattened (batch, sequence, head) row and a vector
    tile of V, eliminating materialized [N, G, F, V] retrieval tensors.
    """
    row = tl.program_id(0)
    v_offsets = tl.program_id(1) * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offsets < V

    score_offsets = tl.arange(0, BLOCK_G)
    score_mask = score_offsets < G
    scores = tl.load(
        selected_scores_ptr + row * G + score_offsets,
        mask=score_mask,
        other=-float("inf"),
    )
    score_max = tl.max(scores, axis=0)
    normalizer = tl.sum(tl.exp(scores - score_max), axis=0)

    head = row % H
    accumulator = tl.zeros((BLOCK_V,), dtype=tl.float32)
    for g in tl.static_range(0, G):
        memory_base = selected_memory_ptr + (row * G + g) * 2
        weight = tl.exp(tl.load(selected_scores_ptr + row * G + g) - score_max) / normalizer
        first_index = tl.load(memory_base)
        second_index = tl.load(memory_base + 1)
        first_ptr = values_ptr + ((head * M + first_index) * 2) * V + v_offsets
        second_ptr = values_ptr + ((head * M + second_index) * 2 + 1) * V + v_offsets
        first_value = tl.load(first_ptr, mask=v_mask, other=0.0)
        second_value = tl.load(second_ptr, mask=v_mask, other=0.0)
        accumulator += weight * (first_value + second_value)

    tl.store(output_ptr + row * V + v_offsets, accumulator, mask=v_mask)


@triton.autotune(configs=_HIERARCHICAL_TOPK_CONFIGS, key=["M", "K", "BLOCK_M"])
@triton.jit
def pkm_local_topk_kernel(
    scores_ptr,
    local_scores_ptr,
    local_indices_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Select exact top-K candidates from one score tile per program."""
    row = tl.program_id(0)
    tile = tl.program_id(1)
    local_offsets = tl.arange(0, BLOCK_M)
    offsets = tile * BLOCK_M + local_offsets
    values = tl.load(scores_ptr + row * M + offsets, mask=offsets < M, other=-float("inf"))
    output_base = (row * tl.cdiv(M, BLOCK_M) + tile) * K
    for rank in tl.static_range(0, K):
        best_value, best_offset = tl.max(values, axis=0, return_indices=True)
        tl.store(local_scores_ptr + output_base + rank, best_value)
        tl.store(local_indices_ptr + output_base + rank, tile * BLOCK_M + best_offset)
        values = tl.where(local_offsets == best_offset, -float("inf"), values)


def fused_factor_scoring(
    query: torch.Tensor,
    keys: torch.Tensor,
    topk: int,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused factor scoring + topk.

    Uses Triton for the einsum computation, then torch.topk for selection.

    Args:
        query: [B, S, F, D] reshaped query
        keys: [F, M, D] factor keys
        topk: number of top indices per factor
        scale: optional scale factor

    Returns:
        topk_scores, topk_indices (long)
    """
    B, S, F, D = query.shape
    M = keys.shape[1]
    device = query.device

    if device.type != "cuda":
        scores = torch.einsum("bsfd,fmd->bsfm", query, keys)
        if scale is not None:
            scores = scores * scale
        return torch.topk(scores, k=topk, dim=-1)

    scores = torch.empty(B, S, F, M, device=device, dtype=query.dtype)

    scale_val = scale if scale is not None else 1.0
    BLOCK_D = max(triton.next_power_of_2(D), 16)
    BLOCK_D = triton.next_power_of_2(D)

    grid = lambda meta: (B * S * F, triton.cdiv(M, meta["BLOCK_M"]))
    pkm_score_kernel[grid](
        query,
        keys,
        scores,
        scale_val,
        B=B, S=S, F=F, D=D, M=M,
        BLOCK_D=BLOCK_D,
    )

    topk_scores, topk_indices = torch.topk(scores, k=topk, dim=-1)
    return topk_scores, topk_indices.to(torch.long)


def fused_topk(scores: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Topk using torch (fused kernel for standalone topk not yet ready).

    Args:
        scores: [..., M] tensor
        topk: number of top indices

    Returns:
        topk_scores, topk_indices (long)
    """
    return torch.topk(scores, k=topk, dim=-1)


def fused_value_aggregate_f2(
    selected_scores: torch.Tensor,
    selected_memory: torch.Tensor,
    values: torch.Tensor,
    heads: int,
    memory_size: int,
) -> torch.Tensor:
    """Return F=2 weighted PKM values without materializing gathered values.

    Args:
        selected_scores: [N, G] candidate scores.
        selected_memory: [N, G, 2] memory indices.
        values: [H, M, 2, V] value table.
        heads: number of retrieval heads.
        memory_size: slots per head and factor.
    """
    if (
        selected_scores.device.type != "cuda"
        or selected_memory.device != selected_scores.device
        or values.device != selected_scores.device
        or selected_memory.shape[-1] != 2
    ):
        raise ValueError("fused F=2 retrieval requires CUDA tensors with two factors")

    rows, topk = selected_scores.shape
    value_dim = values.shape[-1]
    output = torch.empty(rows, value_dim, device=selected_scores.device, dtype=selected_scores.dtype)
    block_g = triton.next_power_of_2(topk)
    grid = lambda meta: (rows, triton.cdiv(value_dim, meta["BLOCK_V"]))
    pkm_retrieve_aggregate_f2_kernel[grid](
        selected_scores,
        selected_memory,
        values,
        output,
        H=heads,
        M=memory_size,
        G=topk,
        V=value_dim,
        BLOCK_G=block_g,
    )
    return output


def hierarchical_topk(
    scores: torch.Tensor,
    topk: int,
    *,
    block_m: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact two-stage top-k for contiguous CUDA score rows.

    A single Triton launch computes local candidates for every (row, tile)
    pair. One PyTorch/CUB merge over only ``ceil(M / block_m) * topk`` values
    returns the exact global top-k. This is exact because a global top-k value
    must be in the local top-k of the tile containing it.
    """
    if scores.device.type != "cuda":
        return torch.topk(scores, k=topk, dim=-1)
    if scores.ndim < 1:
        raise ValueError("scores must have at least one dimension")
    if block_m not in {64, 128, 256}:
        raise ValueError("block_m must be one of 64, 128, or 256")

    memory_size = scores.shape[-1]
    if not 0 < topk <= memory_size:
        raise ValueError("topk must be in [1, scores.shape[-1]]")
    if topk > block_m:
        raise ValueError("hierarchical topk requires topk <= block_m")

    flat_scores = scores.reshape(-1, memory_size)
    if not flat_scores.is_contiguous():
        flat_scores = flat_scores.contiguous()
    rows = flat_scores.shape[0]
    tiles = triton.cdiv(memory_size, block_m)
    local_shape = (rows, tiles, topk)
    local_scores = torch.empty(local_shape, device=scores.device, dtype=scores.dtype)
    local_indices = torch.empty(local_shape, device=scores.device, dtype=torch.long)
    pkm_local_topk_kernel[(rows, tiles)](
        flat_scores,
        local_scores,
        local_indices,
        M=memory_size,
        K=topk,
        BLOCK_M=block_m,
    )

    candidate_scores = local_scores.flatten(1)
    candidate_indices = local_indices.flatten(1)
    selected_scores, candidate_positions = torch.topk(candidate_scores, k=topk, dim=-1)
    selected_indices = torch.gather(candidate_indices, 1, candidate_positions)
    return (
        selected_scores.reshape(*scores.shape[:-1], topk),
        selected_indices.reshape(*scores.shape[:-1], topk),
    )


def available() -> bool:
    """Check if fused PKM kernels are available."""
    return torch.cuda.is_available() and triton is not None
