"""Fused Triton kernels for PKM acceleration.

These kernels provide fast implementations of factor-scoring, top-k selection,
and value retrieval to maximize performance for WideNDepth/sparse-attention models.

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


@triton.autotune(configs=_SCORE_CONFIGS, key=["B", "S", "F", "D", "M"])
@triton.jit
def pkm_score_kernel(
    query_ptr,
    keys_ptr,
    scores_ptr,
    scale_val,  # Passed as runtime scalar to avoid excessive specialization compile variants
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Computes the full score matrix for PKM factor scoring.
    For each (b, s, f), computes scores against all M memory slots.
    """
    off_bsf = tl.program_id(0)
    off_m_block = tl.program_id(1)
    off_b = off_bsf // (S * F)
    off_s = (off_bsf // F) % S
    off_f = off_bsf % F

    q_base = query_ptr + off_b * S * F * D + off_s * F * D + off_f * D
    k_base = keys_ptr + off_f * M * D
    scores_base = scores_ptr + off_b * S * F * M + off_s * F * M + off_f * M

    d_idx = tl.arange(0, BLOCK_D)
    q_mask = d_idx < D
    q = tl.load(q_base + d_idx, mask=q_mask, other=0.0).to(tl.float32)

    m_idx = off_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_idx < M
    d_idx2 = tl.arange(0, BLOCK_D)

    k_ptrs = k_base + m_idx[:, None] * D + d_idx2[None, :]
    k_mask = m_mask[:, None] & (d_idx2[None, :] < D)
    keys = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

    scores = tl.sum(q[None, :] * keys, axis=1) * scale_val

    tl.store(scores_base + m_idx, scores, mask=m_mask)


def fused_factor_scores(query: torch.Tensor, keys: torch.Tensor, scale_val: float) -> torch.Tensor:
    """Computes factor scores across the memory bank.

    Returns the [B, S, F, M] score tensor. Does NOT perform top-K selection.
    """
    if query.ndim != 4:
        raise ValueError(f"query must be 4D, got {query.ndim}D")
    if keys.ndim != 3:
        raise ValueError(f"keys must be 3D, got {keys.ndim}D")

    # Validate device uniformity before fallback routing
    if query.device != keys.device:
        raise ValueError(f"query and keys must be on the same device, got {query.device} and {keys.device}")

    # Enforce strict dtype match and supported precision classes
    valid_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if query.dtype not in valid_dtypes or keys.dtype not in valid_dtypes:
        raise TypeError(f"Dtypes must be FP16, BF16, or FP32. Got query={query.dtype}, keys={keys.dtype}")
    if query.dtype != keys.dtype:
        raise TypeError(f"query and keys must have matching dtypes, got {query.dtype} and {keys.dtype}")

    B, S, F, D = query.shape
    F_k, M, D_k = keys.shape

    if F != F_k or D != D_k:
        raise ValueError(f"Shape mismatch: query(F={F}, D={D}) vs keys(F={F_k}, D={D_k})")

    # Native PyTorch fallback for CPU execution or active autograd backward pass
    needs_grad = torch.is_grad_enabled() and (query.requires_grad or keys.requires_grad)
    if needs_grad or query.device.type != "cuda":
        res = torch.einsum('bsfd,fmd->bsfm', query, keys) * scale_val
        return res.to(query.dtype)

    query = query.contiguous()
    keys = keys.contiguous()

    scores = torch.empty((B, S, F, M), dtype=query.dtype, device=query.device)

    grid = lambda meta: (B * S * F, triton.cdiv(M, meta['BLOCK_M']))
    BLOCK_D = triton.next_power_of_2(D)

    pkm_score_kernel[grid](
        query, keys, scores, float(scale_val),
        B, S, F, D, M,
        BLOCK_D=BLOCK_D
    )

    return scores


@triton.jit
def pkm_local_topk_kernel(
    scores_ptr,
    topk_scores_ptr,
    topk_indices_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Row-wise Top-K selection mapping logic."""
    row = tl.program_id(0)

    scores_base = scores_ptr + row * M
    out_scores_base = topk_scores_ptr + row * K
    out_indices_base = topk_indices_ptr + row * K

    m_idx = tl.arange(0, BLOCK_M)
    m_mask = m_idx < M

    scores = tl.load(scores_base + m_idx, mask=m_mask, other=-float("inf"))

    # Sanitize NaN values to -inf to prevent argmax pollution
    scores = tl.where(scores == scores, scores, -float("inf"))

    is_valid = tl.where(m_mask, 1, 0)

    for k in tl.static_range(0, K):
        valid_scores = tl.where(is_valid == 1, scores, -float("inf"))

        best_idx = tl.argmax(valid_scores, axis=0)

        best_val = tl.max(tl.where(m_idx == best_idx, valid_scores, -float("inf")), axis=0)

        valid_candidate = tl.max(tl.where(m_idx == best_idx, is_valid, 0), axis=0)

        final_val = tl.where(valid_candidate == 1, best_val, -float("inf"))
        final_idx = tl.where(valid_candidate == 1, best_idx, M)

        tl.store(out_scores_base + k, final_val)
        tl.store(out_indices_base + k, final_idx)

        is_valid = tl.where(m_idx == best_idx, 0, is_valid)


def hierarchical_topk(scores: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Python wrapper for Top-K dispatching."""
    if scores.ndim == 0:
        raise ValueError("scores must have at least one dimension")
    if not isinstance(K, int):
        raise TypeError(f"K must be an int, got {type(K)}")

    M = scores.shape[-1]

    if K < 0 or K > M:
        raise ValueError(f"K must satisfy 0 <= K <= M; got K={K}, M={M}")

    if K == 0 or M == 0:
        shape = (*scores.shape[:-1], 0)
        return (
            torch.empty(shape, dtype=scores.dtype, device=scores.device),
            torch.empty(shape, dtype=torch.int64, device=scores.device),
        )

    # Route through torch.topk to retain gradient history or for CPU/large memory regimes
    needs_grad = torch.is_grad_enabled() and scores.requires_grad
    if needs_grad or scores.device.type != "cuda" or K > 32 or M > 2048:
        return torch.topk(scores, K, dim=-1)

    scores_flat = scores.contiguous().view(-1, M)
    N = scores_flat.shape[0]

    topk_scores = torch.empty((N, K), dtype=scores.dtype, device=scores.device)
    topk_indices = torch.empty((N, K), dtype=torch.int64, device=scores.device)

    BLOCK_M = triton.next_power_of_2(M)
    grid = (N,)

    pkm_local_topk_kernel[grid](
        scores_flat, topk_scores, topk_indices,
        M, K, BLOCK_M=BLOCK_M
    )

    return topk_scores.view(*scores.shape[:-1], K), topk_indices.view(*scores.shape[:-1], K)


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
    """Fuse F=2 softmax, value lookup, and weighted aggregation."""
    row = tl.program_id(0)
    v_offsets = tl.program_id(1) * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offsets < V

    head = row % H

    score_offsets = tl.arange(0, BLOCK_G)
    score_mask = score_offsets < G

    scores = tl.load(
        selected_scores_ptr + row * G + score_offsets,
        mask=score_mask,
        other=-float("inf"),
    ).to(tl.float32)

    has_valid = tl.max(scores > -float("inf"), axis=0)
    score_max = tl.max(scores, axis=0)

    shifted_scores = tl.where(scores == -float("inf"), -float("inf"), scores - score_max)
    exps = tl.exp(shifted_scores)

    valid_exps = tl.where(score_mask & (scores > -float("inf")), exps, 0.0)
    normalizer = tl.sum(valid_exps, axis=0)

    normalizer = tl.where(normalizer == 0.0, 1.0, normalizer)

    accumulator = tl.zeros((BLOCK_V,), dtype=tl.float32)

    for g in range(0, G):
        exp_val = tl.sum(tl.where(score_offsets == g, exps, 0.0), axis=0)
        weight = tl.where(has_valid, exp_val / normalizer, 0.0)

        # Load indices and strictly clamp/mask out-of-bounds/sentinel entries before constructing pointers
        first_idx = tl.load(selected_memory_ptr + row * G * 2 + g * 2).to(tl.int64)
        second_idx = tl.load(selected_memory_ptr + row * G * 2 + g * 2 + 1).to(tl.int64)

        first_valid = has_valid & (first_idx >= 0) & (first_idx < M)
        second_valid = has_valid & (second_idx >= 0) & (second_idx < M)

        safe_first_idx = tl.where(first_valid, first_idx, 0)
        safe_second_idx = tl.where(second_valid, second_idx, 0)

        first_ptr = values_ptr + ((head * M + safe_first_idx) * 2) * V + v_offsets
        second_ptr = values_ptr + ((head * M + safe_second_idx) * 2 + 1) * V + v_offsets

        first_value = tl.load(first_ptr, mask=v_mask & first_valid, other=0.0).to(tl.float32)
        second_value = tl.load(second_ptr, mask=v_mask & second_valid, other=0.0).to(tl.float32)

        accumulator += weight * (first_value + second_value)

    out_ptr = output_ptr + row * V + v_offsets
    tl.store(out_ptr, accumulator, mask=v_mask)


def fused_value_aggregate_f2(selected_scores: torch.Tensor, selected_memory: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Python wrapper for F=2 value aggregation with strict validation."""
    if selected_scores.device.type != "cuda":
        raise ValueError("fused_value_aggregate_f2 requires CUDA tensors")

    if (selected_scores.device != selected_memory.device or 
        selected_scores.device != values.device):
        raise ValueError(
            f"all retrieval tensors must share the same CUDA device; got {selected_scores.device}, {selected_memory.device}, {values.device}"
        )

    valid_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if selected_scores.dtype not in valid_dtypes:
        raise TypeError(f"selected_scores dtype must be FP16, BF16, or FP32, got {selected_scores.dtype}")
    if values.dtype not in valid_dtypes:
        raise TypeError(f"values dtype must be FP16, BF16, or FP32, got {values.dtype}")
    if selected_memory.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"selected_memory must be int32 or int64, got {selected_memory.dtype}")

    if selected_scores.ndim != 2:
        raise ValueError(f"selected_scores must be 2D [N, G], got {selected_scores.ndim}D")
    if selected_memory.ndim != 3 or selected_memory.shape[-1] != 2:
        raise ValueError(f"selected_memory must be 3D [N, G, 2], got {selected_memory.shape}")
    if values.ndim != 4 or values.shape[2] != 2:
        raise ValueError(f"values must be 4D [H, M, 2, V], got {values.shape}")

    N, G = selected_scores.shape
    H, M, _, V = values.shape

    if selected_memory.shape[0] != N or selected_memory.shape[1] != G:
        raise ValueError("selected_scores and selected_memory batch/group sizes do not match")
    if H == 0 or M == 0 or V == 0:
        raise ValueError(f"Dimensions H({H}), M({M}), and V({V}) must be > 0")

    if N % H != 0:
        raise ValueError(f"Rows N ({N}) must be divisible by heads H ({H}) to honor the head-mapping contract.")

    selected_scores = selected_scores.contiguous()
    selected_memory = selected_memory.contiguous()
    values = values.contiguous()

    output_dtype = torch.promote_types(selected_scores.dtype, values.dtype)
    output = torch.empty((N, V), dtype=output_dtype, device=values.device)

    if G == 0:
        output.zero_()
        return output

    grid = lambda meta: (N, triton.cdiv(V, meta['BLOCK_V']))
    BLOCK_G = triton.next_power_of_2(G)

    pkm_retrieve_aggregate_f2_kernel[grid](
        selected_scores, selected_memory, values, output,
        H, M, G, V,
        BLOCK_G=BLOCK_G
    )

    return output

def available() -> bool:
    """Check if fused PKM kernels are available."""
    return torch.cuda.is_available() and triton is not None
