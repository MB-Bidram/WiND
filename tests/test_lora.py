"""Architecture-aware LoRA integration tests."""

import copy

import pytest
import torch
from torch import nn

import wind
from wind.language import LMConfig, LMTrainer, LanguageModel


def _config():
    return LMConfig(
        vocab_size=32, dim=16, heads=4, encoder_depth=1, depth=1,
        iterations=1, decoder_depth=1, state_tokens=2, bank_tokens=3,
        max_source_length=8, max_target_length=8, mlp_ratio=2.0,
        wide_type="pkm", pkm_memory_size=16, pkm_heads=2,
        pkm_num_factors=2, pkm_topk=2, pkm_topk_per_factor=4,
        use_alpha_learning=True,
    )


def _inputs(device="cpu"):
    return (torch.tensor([[3, 4, 5, 6]], device=device),
            torch.tensor([[7, 8, 9]], device=device))


def test_discovery_covers_architectural_regions():
    targets = wind.discover_lora_targets(LanguageModel(_config()))
    regions = {target.region for target in targets if target.supported}
    assert {"embeddings", "pkm", "encoder", "compressor", "bank", "depth", "alpha", "decoder", "lm_head"} <= regions
    assert any(target.name == "wide.pkm.keys" and target.kind == "pkm_keys" for target in targets)
    assert any(target.name == "wide.pkm.values" and target.kind == "pkm_values" for target in targets)


def test_zero_lora_preserves_forward_and_regions_are_selected():
    torch.manual_seed(1)
    model = LanguageModel(_config()).eval(); source, labels = _inputs()
    with torch.inference_mode(): reference = model(input_ids=source, labels=labels).logits
    applied = wind.inject_lora(model, wind.LoRAConfig(
        rank=2, regions=("pkm", "compressor", "bank", "depth", "alpha", "decoder", "lm_head", "embeddings"),
    ))
    assert {x.region for x in applied} >= {"pkm", "compressor", "bank", "depth", "alpha", "decoder", "lm_head", "embeddings"}
    with torch.inference_mode(): actual = model(input_ids=source, labels=labels).logits
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    report = wind.lora_parameter_report(model)
    assert report["trainable_parameters"] < report["total_parameters"]
    assert "pkm" in report["regions"] and "depth" in report["regions"]


def test_gradients_optimizer_and_trainer_parameter_selection():
    model = LanguageModel(_config())
    wind.inject_lora(model, wind.LoRAConfig(rank=2, regions=("pkm", "alpha", "depth", "decoder")))
    source, labels = _inputs(); output = model(input_ids=source, labels=labels); output.loss.backward()
    assert any(p.grad is not None for p in wind.lora_parameters(model))
    assert all(not p.requires_grad for name, p in model.named_parameters() if "lora_" not in name and "bias" not in name)
    optimizer = torch.optim.AdamW(wind.lora_parameters(model), lr=1e-2)
    before = {name: p.detach().clone() for name, p in model.named_parameters() if "lora_" in name}
    optimizer.step()
    assert any(not torch.equal(before[name], p) for name, p in model.named_parameters() if name in before)
    trainer = LMTrainer(model, amp=False)
    assert all(p.requires_grad for group in trainer.optimizer.param_groups for p in group["params"])


def test_adapter_checkpoint_and_merge_roundtrip(tmp_path):
    torch.manual_seed(3); source, labels = _inputs()
    base = LanguageModel(_config()); state = copy.deepcopy(base.state_dict())
    cfg = wind.LoRAConfig(rank=2, regions=("pkm", "alpha", "lm_head"))
    wind.inject_lora(base, cfg)
    with torch.no_grad():
        for name, p in base.named_parameters():
            if "lora_B" in name or "lora_delta" in name: p.add_(0.01)
    output = base(input_ids=source, labels=labels).logits.detach()
    path = tmp_path / "adapter.pt"; wind.save_lora_adapters(base, path, cfg)
    loaded = LanguageModel(_config()); loaded.load_state_dict(state); wind.inject_lora(loaded, cfg); wind.load_lora_adapters(loaded, path)
    torch.testing.assert_close(loaded(input_ids=source, labels=labels).logits, output)
    wind.merge_lora(base); merged = base(input_ids=source, labels=labels).logits.detach()
    torch.testing.assert_close(merged, output)
    wind.unmerge_lora(base); torch.testing.assert_close(base(input_ids=source, labels=labels).logits, output)


def test_nested_shared_linear_is_wrapped_once_and_preserves_aliasing():
    class Shared(nn.Module):
        def __init__(self):
            super().__init__(); linear = nn.Linear(4, 4); self.first = linear; self.second = linear
        def forward(self, x): return self.first(x) + self.second(x)
    model = Shared(); wind.inject_lora(model, wind.LoRAConfig(rank=2, include=("first",)))
    assert isinstance(model.first, wind.LoRALinear) and model.first is model.second
    x = torch.randn(2, 4); torch.testing.assert_close(model(x), 2 * model.first.base(x))


def test_dora_zero_delta_gradient_and_merge_roundtrip():
    torch.manual_seed(12)
    model = nn.Sequential(nn.Linear(4, 6), nn.SiLU(), nn.Linear(6, 3))
    reference = copy.deepcopy(model)
    wind.inject_lora(model, wind.LoRAConfig(rank=2, adapter="dora", include=("0", "2")))
    assert isinstance(model[0], wind.DoRALinear) and isinstance(model[2], wind.DoRALinear)
    x = torch.randn(3, 4)
    torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)
    model(x).square().mean().backward()
    assert model[0].lora_magnitude.grad is not None and model[0].lora_A.weight.grad is not None
    with torch.no_grad(): model[0].lora_B.weight.add_(0.05)
    adapted = model(x).detach(); wind.merge_lora(model)
    torch.testing.assert_close(model(x), adapted)
    wind.unmerge_lora(model); torch.testing.assert_close(model(x), adapted)


def test_dora_adapter_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(15); x = torch.randn(2, 4)
    base = nn.Sequential(nn.Linear(4, 4)); state = copy.deepcopy(base.state_dict())
    config = wind.LoRAConfig(rank=2, adapter="dora", include=("0",))
    wind.inject_lora(base, config)
    with torch.no_grad():
        base[0].lora_B.weight.normal_(std=0.02)
    expected = base(x).detach(); path = tmp_path / "dora.pt"
    wind.save_lora_adapters(base, path, config)
    loaded = nn.Sequential(nn.Linear(4, 4)); loaded.load_state_dict(state)
    wind.inject_lora(loaded, config); wind.load_lora_adapters(loaded, path)
    torch.testing.assert_close(loaded(x), expected)


def test_pkm_value_lora_is_selected_slot_adapter_and_preserves_zero_delta():
    torch.manual_seed(13)
    model = LanguageModel(_config()).eval(); source, labels = _inputs()
    with torch.inference_mode(): reference = model(input_ids=source, labels=labels).logits
    applied = wind.inject_lora(model, wind.LoRAConfig(
        rank=2, adapter="pkm_value", regions=("pkm",),
    ))
    assert [target.name for target in applied] == ["wide.pkm.values"]
    adapter = model.wide.pkm._wind_lora_parameter_adapters["p_values"]
    assert isinstance(adapter, wind.PKMValueLoRA)
    with torch.inference_mode(): torch.testing.assert_close(model(input_ids=source, labels=labels).logits, reference, rtol=0, atol=0)
    output = model(input_ids=source, labels=labels); output.loss.backward()
    assert adapter.lora_A.grad is not None and adapter.lora_B.grad is not None
    wind.merge_lora(model); merged = model(input_ids=source, labels=labels).logits.detach()
    wind.unmerge_lora(model); torch.testing.assert_close(model(input_ids=source, labels=labels).logits, merged)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_complete_model_compile_matches_eager_with_lora():
    torch.manual_seed(4); device = "cuda"; model = LanguageModel(_config()).to(device).eval()
    wind.inject_lora(model, wind.LoRAConfig(rank=2, regions=("pkm", "alpha", "depth", "decoder", "lm_head")))
    source, labels = _inputs(device); model.validate_inputs(source, labels=labels)
    with torch.inference_mode(): eager = model(input_ids=source, labels=labels)
    compiled = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    with torch.inference_mode(): actual = compiled(input_ids=source, labels=labels)
    torch.testing.assert_close(actual.logits, eager.logits, atol=3e-4, rtol=6e-3)
    torch.testing.assert_close(actual.loss, eager.loss, atol=3e-4, rtol=6e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_complete_model_compile_matches_eager_with_dora_and_selected_pkm_values():
    torch.manual_seed(14); device = "cuda"; source, labels = _inputs(device)
    model = LanguageModel(_config()).to(device).eval()
    wind.inject_lora(model, wind.LoRAConfig(
        rank=2, adapter="dora", regions=("decoder", "lm_head"),
        overrides=({"adapter": "pkm_value", "regions": ("pkm",)},),
    ))
    model.validate_inputs(source, labels=labels)
    with torch.inference_mode(): eager = model(input_ids=source, labels=labels)
    compiled = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    with torch.inference_mode(): actual = compiled(input_ids=source, labels=labels)
    torch.testing.assert_close(actual.logits, eager.logits, atol=3e-4, rtol=6e-3)
    torch.testing.assert_close(actual.loss, eager.loss, atol=3e-4, rtol=6e-3)
