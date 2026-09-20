"""WideNDepth data-flow components.

The architecture intentionally keeps knowledge production and reasoning
separate:

``input -> Wide -> Encoder -> Compressor -> read-only FeatureBank``
``                                  -> iterative Depth + bounded Retrieval -> output``
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .attention import Attention
from .modules import Depth, WindModule, RMSNorm


def _resolve_norm_instance(norm: str, dim: int):
    """Return the norm layer specified by *norm*."""

    if norm == "rmsnorm":
        return RMSNorm(dim)
    if norm == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"unknown norm: {norm!r}")


class LearnedQueryCompressor(WindModule):
    """Learned-query attention pooling over source tokens.

    Unlike :class:`Compressor` (which uses adaptive average pooling), this
    module learns a fixed set of query vectors that attend to the full source
    sequence, producing ``count`` output tokens.  An optional ``mask`` argument
    restricts attention to valid (non-padding) positions.
    """

    def __init__(self, dim: int, count: int, *, heads: int = 8,
                 dropout: float = 0.0, use_rope: bool = True,
                 rope_theta: float = 10000.0, rope_max_seq_len: int = 2048,
                 norm: str = "rmsnorm"):
        super().__init__()
        if count < 1:
            raise ValueError("count must be positive")
        self.dim, self.count = dim, count
        self.queries = nn.Parameter(torch.randn(count, dim) * 0.02)
        self.attn = Attention(dim, heads, dropout, use_rope, rope_theta,
                              rope_max_seq_len)
        self.norm = _resolve_norm_instance(norm, dim)

    def forward(self, source: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        if source.ndim != 3 or source.size(-1) != self.dim:
            raise ValueError(f"expected [batch, sequence, {self.dim}]")
        queries = self.queries.unsqueeze(0).expand(source.size(0), -1, -1)
        pooled, _ = self.attn(queries, source, mask=mask)
        return self.norm(queries + pooled)


class Compressor(WindModule):
    """Compress encoder tokens before they enter the knowledge bank."""

    def __init__(self, dim: int, tokens: int = 64):
        super().__init__()
        if tokens < 1:
            raise ValueError("tokens must be positive")
        self.dim, self.tokens = dim, tokens
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.dim:
            raise ValueError(f"expected [batch, sequence, {self.dim}]")
        x = self.proj(self.norm(x))
        if x.size(1) <= self.tokens:
            return x
        # Adaptive pooling is differentiable and handles non-divisible lengths.
        return F.adaptive_avg_pool1d(x.transpose(1, 2), self.tokens).transpose(1, 2)


class FeatureBank(WindModule):
    """Read-only bank interface.

    ``forward`` creates a bank tensor; ``read`` is the only operation exposed
    to the reasoning path. No mutable cache is kept on the module, so separate
    requests cannot leak knowledge into one another.
    """

    def __init__(self, dim: int, max_tokens: int = 64, detach: bool = True):
        super().__init__()
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.dim, self.max_tokens, self.detach = dim, max_tokens, detach
        self.norm = nn.LayerNorm(dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.size(-1) != self.dim:
            raise ValueError(f"expected [batch, sequence, {self.dim}]")
        selected = features[:, : self.max_tokens]
        if self.detach:
            # The default bank is a read-only inference boundary. Avoid
            # constructing a throwaway autograd graph for its normalization.
            with torch.no_grad():
                return self.norm(selected)
        return self.norm(selected)

    def read(self, bank: torch.Tensor, limit: int | None = None) -> torch.Tensor:
        """Return at most ``limit`` tokens, enforcing the bank read boundary."""

        limit = self.max_tokens if limit is None else min(limit, self.max_tokens)
        if limit < 1:
            raise ValueError("read limit must be positive")
        return bank[:, :limit]


class AdaptiveFeatureBank(FeatureBank):
    """Importance-ranked bank with a hard token budget."""
    def __init__(self, dim: int, max_tokens: int = 64, detach: bool = True):
        super().__init__(dim, max_tokens, detach); self.score = nn.Linear(dim, 1)
    def forward(self, features):
        if features.size(1) <= self.max_tokens: return super().forward(features)
        if self.detach:
            with torch.no_grad():
                idx = self.score(features).squeeze(-1).topk(self.max_tokens, dim=1).indices
                selected = features.gather(1, idx.unsqueeze(-1).expand(-1, -1, features.size(-1)))
        else:
            idx = self.score(features).squeeze(-1).topk(self.max_tokens, dim=1).indices
            selected = features.gather(1, idx.unsqueeze(-1).expand(-1, -1, features.size(-1)))
        return super().forward(selected)


class Retrieval(WindModule):
    """Bounded cross-attention from reasoning tokens into the bank."""

    def __init__(self, dim: int, read_tokens: int = 8):
        super().__init__()
        if read_tokens < 1:
            raise ValueError("read_tokens must be positive")
        self.dim, self.read_tokens = dim, read_tokens
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def prepare_bank(self, bank: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project a bank once; the result can be reused across iterations."""

        return self.k_proj(bank), self.v_proj(bank)

    def forward(
        self,
        query: torch.Tensor,
        bank: torch.Tensor | None = None,
        *,
        bank_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if bank_cache is None and bank is None:
            raise ValueError("provide bank or bank_cache")
        if query.ndim != 3 or query.size(-1) != self.dim:
            raise ValueError(f"expected query with final dimension {self.dim}")
        if bank is not None and (bank.ndim != 3 or bank.size(-1) != self.dim):
            raise ValueError(f"expected bank with final dimension {self.dim}")
        q = self.q_proj(query)
        if bank_cache is None:
            k, v = self.prepare_bank(bank)
        else:
            k, v = bank_cache
            if k.ndim != 3 or v.ndim != 3:
                raise ValueError("bank_cache must contain [batch, tokens, dim] tensors")
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)
        count = min(self.read_tokens, k.size(1))
        indices = scores.topk(count, dim=-1).indices
        selected_v = v.unsqueeze(1).expand(-1, query.size(1), -1, -1).gather(
            2, indices.unsqueeze(-1).expand(-1, -1, -1, self.dim)
        )
        weights = torch.softmax(torch.gather(scores, -1, indices), dim=-1)
        return self.out_proj((weights.unsqueeze(-1) * selected_v).sum(dim=-2))


class ReasoningDepth(WindModule):
    """Iterative depth stack with bounded, read-only bank retrieval."""

    def __init__(self, layers: nn.Module | list[nn.Module] | tuple[nn.Module, ...],
                 dim: int, iterations: int = 1, read_tokens: int = 8):
        super().__init__()
        if iterations < 1:
            raise ValueError("iterations must be positive")
        self.layers = layers if isinstance(layers, nn.ModuleList) else nn.ModuleList(
            [layers] if isinstance(layers, nn.Module) else list(layers)
        )
        if not self.layers:
            raise ValueError("ReasoningDepth requires at least one layer")
        self.iterations = iterations
        self.retrieval = Retrieval(dim, read_tokens)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        bank_cache = self.retrieval.prepare_bank(bank)
        for index in range(self.iterations):
            layer = self.layers[index % len(self.layers)]
            context = self.retrieval(x, bank_cache=bank_cache)
            x = x + torch.sigmoid(self.gate) * context
            x = layer(x)
        return x


class WideNDepth(WindModule):
    """Reference implementation of the supplied WideNDepth diagram."""

    def __init__(self, wide: nn.Module, encoder: nn.Module, compressor: nn.Module,
                 depth: ReasoningDepth | nn.Module | list[nn.Module] | tuple[nn.Module, ...],
                 *, dim: int, bank_tokens: int = 64, read_tokens: int = 8,
                 iterations: int = 1, detach_bank: bool = True,
                 bank: nn.Module | None = None):
        super().__init__()
        self.wide = wide
        self.encoder = encoder
        self.compressor = compressor
        self.bank = bank if bank is not None else FeatureBank(
            dim, bank_tokens, detach=detach_bank
        )
        self.depth = depth if isinstance(depth, ReasoningDepth) else ReasoningDepth(
            depth, dim, iterations=iterations, read_tokens=read_tokens
        )

    def forward(self, x: torch.Tensor, *, return_aux: bool = False):
        branch_outputs = None
        if return_aux and hasattr(self.wide, "forward_with_branches"):
            knowledge, branch_outputs = self.wide.forward_with_branches(x)
        else:
            knowledge = self.wide(x)
        encoded = self.encoder(knowledge)
        # The bank retains rich encoder knowledge; Depth reasons over a
        # separately compressed stream and may only retrieve bounded slices.
        compressed = self.compressor(encoded)
        bank = self.bank(encoded)
        output = self.depth(compressed, bank)
        if return_aux:
            return output, {"knowledge": encoded, "compressed": compressed,
                            "bank": bank, "branches": branch_outputs}
        return output

    def knowledge_parameters(self):
        """Parameters for a standalone Wide/Encoder pre-training stage."""

        yield from self.wide.parameters()
        yield from self.encoder.parameters()
        yield from self.compressor.parameters()
        yield from self.bank.parameters()

    def reasoning_parameters(self):
        """Parameters updated while the knowledge path is frozen."""

        yield from self.depth.parameters()

    def freeze_knowledge(self) -> "WideNDepth":
        """Freeze Wide, Encoder, Compressor and Bank for reasoning training."""

        for parameter in self.knowledge_parameters():
            parameter.requires_grad_(False)
        return self

    def freeze_reasoning(self) -> "WideNDepth":
        for parameter in self.reasoning_parameters():
            parameter.requires_grad_(False)
        return self

    def unfreeze_knowledge(self) -> "WideNDepth":
        for parameter in self.knowledge_parameters():
            parameter.requires_grad_(True)
        return self

    def unfreeze_reasoning(self) -> "WideNDepth":
        for parameter in self.reasoning_parameters():
            parameter.requires_grad_(True)
        return self


class EncoderDecoder(WindModule):
    """Encoder-decoder WND with bounded cross-attention into encoder memory."""

    def __init__(
        self,
        encoder: WideNDepth,
        decoder: ReasoningDepth | nn.Module | list[nn.Module] | tuple[nn.Module, ...],
        *,
        dim: int,
        read_tokens: int = 8,
        iterations: int = 1,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder if isinstance(decoder, ReasoningDepth) else ReasoningDepth(
            decoder, dim, iterations=iterations, read_tokens=read_tokens
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        knowledge = self.encoder.wide(source)
        encoded = self.encoder.encoder(knowledge)
        bank = self.encoder.bank(encoded)
        # Target is already tokenized/projected by the caller.
        return self.decoder(target, bank)




