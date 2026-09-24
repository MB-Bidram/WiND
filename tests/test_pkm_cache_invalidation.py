"""PKM derived inference keys must not survive an optimizer update."""

import torch

from winc.pkm import make_pkm_wide


def test_pkm_inference_key_cache_invalidated_by_training_mode():
    torch.manual_seed(31)
    module = make_pkm_wide(dim=16, memory_size=16, query_dim=8, value_dim=16,
                           num_factors=2, heads=2, topk=2, topk_per_factor=4).eval()
    x = torch.randn(1, 3, 16)
    with torch.inference_mode(): module(x)  # populate derived inference keys
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    module.train()
    optimizer.zero_grad(set_to_none=True)
    module(x).square().mean().backward(); optimizer.step()
    module.eval()
    with torch.inference_mode(): actual = module(x)
    # A freshly loaded copy has no derived cache and is the reference state.
    fresh = make_pkm_wide(dim=16, memory_size=16, query_dim=8, value_dim=16,
                          num_factors=2, heads=2, topk=2, topk_per_factor=4).eval()
    fresh.load_state_dict(module.state_dict())
    with torch.inference_mode(): expected = fresh(x)
    torch.testing.assert_close(actual, expected)
