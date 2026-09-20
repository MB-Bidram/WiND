"""Generation KV cache runtime.

The Engine owns cache mechanics for request-local inference.
This provides a preallocated, append-efficient cache that avoids
the O(sequence_length) reallocations caused by torch.cat per token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class CacheView:
    """A view into a GenerationCache for a specific layer and cache type."""
    cache: "GenerationCache"
    layer_idx: int
    cache_type: str  # "self" or "cross"

    @property
    def available(self) -> bool:
        return self.cache is not None

    def get(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.cache is None:
            return None
        return self.cache.get(self.layer_idx, self.cache_type)

    def append(self, key: torch.Tensor, val: torch.Tensor) -> None:
        if self.cache is not None:
            self.cache.append(self.layer_idx, self.cache_type, key, val)


@dataclass
class LayerCache:
    """Holds self-attention and cross-attention caches for one decoder layer."""
    self_cache: CacheView | None = None
    cross_cache: CacheView | None = None


class GenerationCache:
    """Request-local KV cache with preallocated storage.

    Instead of repeatedly catting, this cache preallocates buffers at
    a configurable capacity and appends in-place up to that capacity.

    Args:
        batch_size: Batch dimension size.
        max_seq_len: Maximum sequence length the cache can hold.
        num_heads: Number of attention heads.
        head_dim: Per-head dimension.
        dtype: Tensor dtype.
        device: Tensor device.
        n_layers: Number of decoder layers.
    """

    def __init__(self, batch_size: int, max_seq_len: int, num_heads: int,
                 head_dim: int, *, dtype=torch.float32, device="cpu", n_layers: int = 1):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.n_layers = n_layers
        # Decoder layers are appended one at a time.  A single global length
        # would make layer 1 observe layer 0's cache position while the first
        # token of a decode step is still being processed.
        self._lengths = [0] * n_layers

        shape = (batch_size, num_heads, max_seq_len, head_dim)
        if n_layers > 1:
            self._key_cache = torch.zeros((n_layers,) + shape, dtype=dtype, device=device)
            self._val_cache = torch.zeros((n_layers,) + shape, dtype=dtype, device=device)
        else:
            self._key_cache = torch.zeros(shape, dtype=dtype, device=device)
            self._val_cache = torch.zeros(shape, dtype=dtype, device=device)

        # Per-layer cross-attention caches (static, set once at start).
        # Each layer may have different K/V projections.
        # Stored as list of (key, val) tuples, indexed by layer_idx.
        self._cross_caches: list[tuple[torch.Tensor, torch.Tensor] | None] = (
            [None] * n_layers
        )

    def set_cross_cache(self, key: torch.Tensor, val: torch.Tensor) -> None:
        """Set the static cross-attention cache for all layers (encoder output K/V).

        Called once at the start of generation; the cache remains fixed.
        """
        for layer_idx in range(self.n_layers):
            self._cross_caches[layer_idx] = (key, val)

    def set_cross_cache_for_layer(self, layer_idx: int, key: torch.Tensor,
                                   val: torch.Tensor) -> None:
        """Set the static cross-attention cache for a specific layer.

        Allows per-layer K/V projections for cross-attention.
        """
        self._cross_caches[layer_idx] = (key, val)

    def append(self, layer_idx: int, cache_type: str, key: torch.Tensor,
               val: torch.Tensor) -> None:
        """Append key/value projections to the self-attention cache.

        Args:
            layer_idx: Layer index.
            cache_type: "self" or "cross".
            key: [batch, heads, new_tokens, head_dim]
            val: [batch, heads, new_tokens, head_dim]
        """
        if cache_type == "cross":
            # Cross cache is set once, not appended
            return

        n = key.size(-2)
        old_len = self._lengths[layer_idx]
        new_len = old_len + n
        if new_len > self.max_seq_len:
            raise ValueError(f"cache overflow: {new_len} > {self.max_seq_len}")

        key = key.to(device=self.device, dtype=self.dtype)
        val = val.to(device=self.device, dtype=self.dtype)
        if self.n_layers > 1:
            self._key_cache[layer_idx, :, :, old_len:new_len] = key
            self._val_cache[layer_idx, :, :, old_len:new_len] = val
        else:
            self._key_cache[:, :, old_len:new_len] = key
            self._val_cache[:, :, old_len:new_len] = val
        self._lengths[layer_idx] = new_len

    def get(self, layer_idx: int, cache_type: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the cached key/value for a layer and cache type."""
        if cache_type == "cross":
            cached = self._cross_caches[layer_idx] if 0 <= layer_idx < len(self._cross_caches) else None
            return cached

        slc = slice(self._lengths[layer_idx])
        if self.n_layers > 1:
            return (self._key_cache[layer_idx, :, :, slc], self._val_cache[layer_idx, :, :, slc])
        return (self._key_cache[:, :, slc], self._val_cache[:, :, slc])

    def get_view(self, layer_idx: int, cache_type: str) -> CacheView:
        """Get a CacheView for a layer/cache_type pair."""
        return CacheView(self, layer_idx, cache_type)

    def reset(self) -> None:
        """Clear the self-attention cache (keeps cross-cache set)."""
        self._lengths = [0] * self.n_layers

    def reset_all(self) -> None:
        """Clear all caches."""
        self._lengths = [0] * self.n_layers
        self._cross_caches = [None] * self.n_layers

    @property
    def length(self) -> int:
        """Length of a completed decode cache.

        During a decoder-layer loop the layers can temporarily differ by one
        token; callers that need a layer-local view must use ``get``.
        """
        return self._lengths[0]

    def to(self, device, dtype=None) -> "GenerationCache":
        """Move cache to a device/dtype."""
        new_cache = GenerationCache(
            self.batch_size, self.max_seq_len, self.num_heads, self.head_dim,
            dtype=dtype or self.dtype, device=device, n_layers=self.n_layers,
        )
        new_cache._lengths = self._lengths.copy()
        if self.n_layers > 1:
            new_cache._key_cache.copy_(
                self._key_cache.to(device=device, dtype=dtype or self.dtype)
            )
            new_cache._val_cache.copy_(
                self._val_cache.to(device=device, dtype=dtype or self.dtype)
            )
        else:
            new_cache._key_cache.copy_(
                self._key_cache.to(device=device, dtype=dtype or self.dtype)
            )
            new_cache._val_cache.copy_(
                self._val_cache.to(device=device, dtype=dtype or self.dtype)
            )
        new_cache._cross_caches = []
        for layer_idx in range(self.n_layers):
            cached = self._cross_caches[layer_idx]
            if cached is not None:
                k, v = cached
                new_cache._cross_caches.append(
                    (k.to(device=device, dtype=dtype or self.dtype), v.to(device=device, dtype=dtype or self.dtype))
                )
            else:
                new_cache._cross_caches.append(None)
        return new_cache
