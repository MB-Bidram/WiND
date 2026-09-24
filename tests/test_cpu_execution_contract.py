"""CPU-only public execution-contract and WiNDBreaker coverage."""

import torch

import wind
from wind.language import LMConfig, LanguageModel
from winc.architecture import AdaptiveFeatureBank
from windbreaker import inspect_model


def test_generic_state_and_bank_budgets_and_cyclic_layer_count_cpu():
    model = wind.build(wind.Config(
        dim=16, heads=4, depth=3, iterations=5,
        state_tokens=4, bank_tokens=6, read_tokens=2,
    ))
    x = torch.randn(2, 8, 16)
    with inspect_model(model) as inspection:
        y = model(x)
        y.square().mean().backward()
    assert y.shape == (2, 4, 16)
    assert [event.module for event in inspection.report.layer_applications] == [
        "depth.layers.0", "depth.layers.1", "depth.layers.2", "depth.layers.0", "depth.layers.1",
    ]
    assert [event.reasoning_pass for event in inspection.report.layer_applications] == list(range(5))
    assert len(inspection.report.bank_reads) == 5
    assert all(not event.properties.differentiable_end_to_end for event in inspection.report.bank_reads)
    assert any(event.gradient_seen for event in inspection.report.layer_applications)
    assert not y.is_cuda


def test_language_full_stack_iteration_count_and_hooks_cpu():
    config = LMConfig(
        vocab_size=32, dim=16, heads=4, width=1, encoder_depth=1,
        depth=3, iterations=2, decoder_depth=1, state_tokens=4, bank_tokens=6,
        max_source_length=8, max_target_length=6, mlp_ratio=2.0,
    )
    model = LanguageModel(config)
    source = torch.randint(3, 32, (2, 6))
    labels = torch.randint(3, 32, (2, 4))
    with inspect_model(model) as inspection:
        output = model(source, labels=labels)
        output.loss.backward()
    assert output.logits.shape == (2, 4, 32)
    assert [event.reasoning_pass for event in inspection.report.layer_applications] == [0, 0, 0, 1, 1, 1]
    assert len(inspection.report.bank_reads) == 6
    assert all(event.properties.differentiable_end_to_end for event in inspection.report.bank_reads)
    assert any(event.gradient_seen for event in inspection.report.layer_applications)
    assert not output.logits.is_cuda


def test_adaptive_feature_bank_scorer_receives_task_gradient_when_differentiable():
    bank = AdaptiveFeatureBank(8, max_tokens=3, detach=False)
    features = torch.randn(2, 6, 8, requires_grad=True)
    bank(features).square().mean().backward()
    assert bank.score.weight.grad is not None
    assert bank.score.weight.grad.abs().sum() > 0
