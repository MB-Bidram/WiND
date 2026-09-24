"""Mathematically exact PKM candidate pruning contracts."""

import torch

from winc.pkm import FactorizedPKM


def _pair(pruned: bool, factors: int = 2):
    module = FactorizedPKM(query_dim=32, memory_size=32, value_dim=8,
                           num_factors=factors, heads=1, topk=4,
                           topk_per_factor=8, exact_candidate_pruning=pruned)
    return module


@torch.no_grad()
def test_exact_candidate_pruning_matches_full_candidates_for_random_scores():
    torch.manual_seed(41)
    baseline, pruned = _pair(False), _pair(True)
    pruned.load_state_dict(baseline.state_dict())
    query = torch.randn(2, 3, 32)
    torch.testing.assert_close(pruned(query), baseline(query), rtol=0, atol=0)


def test_exact_candidate_pruning_preserves_input_and_parameter_gradients():
    torch.manual_seed(42)
    baseline, pruned = _pair(False), _pair(True)
    pruned.load_state_dict(baseline.state_dict())
    q0, q1 = torch.randn(1, 2, 32, requires_grad=True), torch.randn(1, 2, 32, requires_grad=True)
    q1.data.copy_(q0.data)
    baseline(q0).square().mean().backward(); pruned(q1).square().mean().backward()
    torch.testing.assert_close(q0.grad, q1.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(baseline.keys.grad, pruned.keys.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(baseline.values.grad, pruned.values.grad, rtol=1e-5, atol=1e-6)


def test_return_aux_keeps_configured_candidate_count():
    torch.manual_seed(43)
    module = _pair(True)
    _, aux = module(torch.randn(1, 2, 32), return_aux=True)
    assert aux["topk_per_factor"] == 8
