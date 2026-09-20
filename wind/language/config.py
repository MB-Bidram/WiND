"""LMConfig dataclass for LanguageModel configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class LMConfig:
    vocab_size: int
    dim: int = 256
    heads: int = 8
    width: int = 2
    encoder_depth: int = 2
    depth: int = 4
    iterations: int = 1
    decoder_depth: int = 2
    state_tokens: int = 16
    bank_tokens: int = 64
    max_source_length: int = 512
    max_target_length: int = 512
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    checkpointing: bool = False
    norm: str = "rmsnorm"
    use_rope: bool = True
    rope_theta: float = 10000.0
    rope_max_seq_len: int = 2048

    wide_type: str = "dense"
    encoder_type: str = "standard"
    use_alpha_learning: bool = False
    alpha_init: float = 0.0
    custom_modules: dict = field(default_factory=dict)

    pkm_memory_size: int = 4096
    pkm_num_factors: int = 2
    pkm_heads: int = 4
    pkm_topk: int = 32
    pkm_topk_per_factor: int | None = None
    pkm_key_dtype: str = "float32"
    pkm_value_dtype: str = "float32"
    pkm_similarity: str = "cosine"
    pkm_exact_candidate_pruning: bool = False

    def __post_init__(self):
        for name in ("vocab_size", "dim", "heads", "width", "encoder_depth", "depth",
                     "iterations", "decoder_depth", "state_tokens", "bank_tokens",
                     "max_source_length", "max_target_length", "rope_max_seq_len"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if not math.isfinite(self.mlp_ratio) or self.dim * self.mlp_ratio < 1:
            raise ValueError("mlp_ratio must give at least one hidden unit")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.norm not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm must be 'layernorm' or 'rmsnorm'")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        ids = (self.pad_token_id, self.bos_token_id, self.eos_token_id)
        if any(type(i) is not int or not 0 <= i < self.vocab_size for i in ids):
            raise ValueError("special token IDs must be within the vocabulary")
        if len(set(ids)) != 3:
            raise ValueError("pad, bos, and eos IDs must be distinct")
        if self.wide_type not in {"dense", "pkm"}:
            raise ValueError("wide_type must be 'dense' or 'pkm'")
        if self.encoder_type not in {"standard", "small"}:
            raise ValueError("encoder_type must be 'standard' or 'small'")
        if self.pkm_key_dtype not in {"float32", "float16", "bfloat16", "int8"}:
            raise ValueError(f"pkm_key_dtype must be 'float32', 'float16', 'bfloat16', or 'int8', got: {self.pkm_key_dtype}")
        if self.pkm_value_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"pkm_value_dtype must be 'float32', 'float16', or 'bfloat16', got: {self.pkm_value_dtype}")
        if self.pkm_similarity not in {"cosine", "dot"}:
            raise ValueError("pkm_similarity must be 'cosine' or 'dot'")
        if type(self.pkm_exact_candidate_pruning) is not bool:
            raise ValueError("pkm_exact_candidate_pruning must be a bool")


def _map_pkm_dtype(s: str) -> torch.dtype:
    """Map a string dtype to torch.dtype."""
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
    }[s]


def _validate_pkm_config(c: LMConfig) -> None:
    """Validate that PKM configuration parameters are consistent."""
    query_dim = c.dim // 2
    head_dim = query_dim // (c.pkm_heads * c.pkm_num_factors)
    if query_dim != head_dim * c.pkm_heads * c.pkm_num_factors:
        remainder = query_dim % (c.pkm_heads * c.pkm_num_factors)
        raise ValueError(
            f"query_dim ({query_dim}) must be divisible by "
            f"pkm_heads ({c.pkm_heads}) * pkm_num_factors ({c.pkm_num_factors}). "
            f"Remainder: {remainder}. Try adjusting dim, pkm_heads, or pkm_num_factors."
        )


