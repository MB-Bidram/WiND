"""Hot-loop compilation keeps eager input validation available via preflight."""

import pytest
import torch

from wind.language import LMConfig, LanguageModel


def _model():
    return LanguageModel(LMConfig(
        vocab_size=32, dim=16, heads=4, encoder_depth=1, depth=1,
        decoder_depth=1, width=1, iterations=1, state_tokens=2,
        bank_tokens=3, max_source_length=8, max_target_length=8,
        mlp_ratio=2.0, use_alpha_learning=True,
    ))


def test_validate_inputs_retains_out_of_range_error():
    with pytest.raises(ValueError, match="outside the vocabulary"):
        _model().validate_inputs(torch.tensor([[32]]))


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_language_model_fullgraph_eager_backend_after_preflight():
    model = _model().eval()
    source = torch.tensor([[3, 4, 5]])
    labels = torch.tensor([[6, 7]])
    model.validate_inputs(source, labels=labels)
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    with torch.inference_mode():
        result = compiled(input_ids=source, labels=labels)
    assert result.logits.shape == (1, 2, 32)
