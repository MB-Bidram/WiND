"""Optional separation and anti-collapse objectives."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def orthogonality_loss(branches: Tensor, eps: float = 1e-6) -> Tensor:
    """Penalize correlation between Wide branches.

    ``branches`` has shape ``[branches, ...]``. The diagonal is encouraged to
    remain one while off-diagonal correlations approach zero.
    """

    if branches.ndim < 2 or branches.size(0) < 2:
        return branches.new_zeros(())
    z = F.normalize(branches.flatten(1), dim=-1, eps=eps)
    gram = z @ z.transpose(0, 1)
    return (gram - torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)).square().mean()


def diversity_loss(tokens: Tensor, eps: float = 1e-6) -> Tensor:
    """Discourage FeatureBank slots from collapsing to the same vector."""

    if tokens.ndim != 3 or tokens.size(1) < 2:
        return tokens.new_zeros(())
    z = F.normalize(tokens, dim=-1, eps=eps)
    sim = torch.matmul(z, z.transpose(-2, -1))
    eye = torch.eye(tokens.size(1), device=tokens.device, dtype=torch.bool)
    return sim[..., ~eye].square().mean()
