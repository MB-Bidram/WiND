"""Numerical, gradient, and ownership contracts for WiNC bounded retrieval."""

import pytest
import torch

from winc.architecture import Retrieval


def _reference(module: Retrieval, query: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    q = module.q_proj(query); k, v = module.prepare_bank(bank)
    scores = q @ k.transpose(-2, -1) / (module.dim ** 0.5)
    count = min(module.read_tokens, bank.size(1))
    indices = scores.topk(count, dim=-1).indices
    values = v.unsqueeze(1).expand(-1, query.size(1), -1, -1).gather(
        2, indices.unsqueeze(-1).expand(-1, -1, -1, module.dim)
    )
    weights = torch.softmax(scores.gather(-1, indices), dim=-1)
    return module.out_proj((weights.unsqueeze(-1) * values).sum(dim=-2))


@pytest.mark.parametrize("read_tokens,bank_tokens", [(1, 5), (5, 5), (8, 5)])
def test_retrieval_matches_reference_and_preserves_bank(read_tokens, bank_tokens):
    torch.manual_seed(21)
    module = Retrieval(8, read_tokens)
    query = torch.randn(2, 3, 8, requires_grad=True)
    bank = torch.randn(2, bank_tokens, 8, requires_grad=True)
    before = bank.detach().clone()
    actual = module(query, bank)
    expected = _reference(module, query, bank)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(bank, before)
    actual.square().mean().backward()
    assert query.grad is not None and bank.grad is not None
    assert all(p.grad is not None for p in module.parameters())


def test_retrieval_accepts_noncontiguous_query_and_cached_bank():
    torch.manual_seed(22)
    module = Retrieval(8, 3)
    query = torch.randn(2, 8, 3).transpose(1, 2)  # [2, 3, 8], noncontiguous
    bank = torch.randn(2, 5, 8)
    cached = module.prepare_bank(bank)
    torch.testing.assert_close(module(query, bank), module(query, bank_cache=cached))


def test_retrieval_requires_a_bank_source():
    with pytest.raises(ValueError, match="provide bank"):
        Retrieval(8)(torch.randn(1, 2, 8))
