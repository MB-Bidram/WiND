# WiND

WiND is a PyTorch framework for building, training, adapting, and inspecting
WideNDepth models. It provides a compact generic model API, a conditional
language-model API, training utilities, `torch.compile` integration, and
architecture-level runtime inspection.

For the WideNDepth architecture and its research context, see
[MB-Bidram/WideNDepth](https://github.com/MB-Bidram/WideNDepth).

## Installation

WiND requires Python 3.10+ and PyTorch 2.0+.

```bash
pip install .
```

For development:

```bash
pip install -e .
python -m pytest -q
```

## Quick start: generic model

`wind.build()` creates a tensor-to-state model. This dependency-free CPU
example performs a complete forward and backward pass.

```python
import torch
import wind

model = wind.build(wind.Config(
    dim=16,
    heads=4,
    depth=2,
    iterations=2,
    state_tokens=4,
    bank_tokens=6,
    read_tokens=2,
))

features = torch.randn(2, 8, 16)
state = model(features)
state.square().mean().backward()

print(state.shape)  # torch.Size([2, 4, 16])
```

## Model APIs

WiND supports two public construction paths.

| API | Input | Output | Use case |
| --- | --- | --- | --- |
| `wind.build(Config(...))` | feature tensor `[batch, tokens, dim]` | reasoning state `[batch, state_tokens, dim]` | custom encoders, multimodal inputs, and direct architectural composition |
| `LanguageModel(LMConfig(...))` | source token IDs plus target token IDs | logits, optional loss, and token count | conditional language modelling, training, checkpoints, and generation |

### `depth` and `iterations`

These names have precise meanings in both APIs:

- **depth** is the number of distinct reasoning layers.
- **iterations** is the number of reasoning passes. It is not a layer count.

The execution unit differs by API:

| API | One iteration executes | Layer applications for `depth=6, iterations=1` |
| --- | --- | --- |
| Generic `build()` | one cyclic reasoning-layer application | layer `0` once |
| `LanguageModel` | the complete Depth stack, in order | layers `0` through `5` once each |

Consequently, generic `depth=6, iterations=8` applies layers `0, 1, 2, 3, 4,
5, 0, 1`; a language model with the same values executes all six layers on
each of eight passes, for 48 layer applications.

### State and memory budgets

`state_tokens` and `bank_tokens` are separate configuration values in both
APIs:

- `state_tokens`: number of compressed tokens passed through the reasoning path.
- `bank_tokens`: number of memory tokens available to retrieval.

Set both explicitly when tuning model capacity or memory use.

## Conditional language model

`LanguageModel` is a conditional/prefix language model. Source tokens enter
the knowledge and reasoning path. Target tokens are shifted and decoded
causally, so the source must not include a target continuation during
training.

```python
import torch
from wind import LMConfig, LanguageModel

config = LMConfig(
    vocab_size=32_000,
    dim=256,
    heads=8,
    width=2,
    encoder_depth=2,
    depth=4,
    iterations=2,
    decoder_depth=2,
    state_tokens=16,
    bank_tokens=64,
    max_source_length=256,
    max_target_length=128,
)

model = LanguageModel(config)
source_ids = torch.randint(0, config.vocab_size, (2, 64))
labels = torch.randint(0, config.vocab_size, (2, 32))

output = model(input_ids=source_ids, labels=labels)
output.loss.backward()
print(output.logits.shape)  # torch.Size([2, 32, 32000])
```

`labels` accepts `-100` as an ignored target value. `attention_mask` has
source shape `[batch, source_tokens]`; `True`/`1` marks a valid source token.

## Training and checkpoints

`LMTrainer` handles device transfer, autocast on CUDA, gradient accumulation,
gradient clipping, optimizer construction, and resumable checkpoints. Each
batch is a mapping accepted by `LanguageModel.forward`, usually containing
`input_ids` and `labels`.

```python
from wind import LMTrainer

batches = [{"input_ids": source_ids, "labels": labels}]

trainer = LMTrainer(model, lr=3e-4, grad_accumulation=1)
loss = trainer.step(batches)
trainer.save_checkpoint("training.pt")

model.save("model.pt")
reloaded = LanguageModel.load("model.pt", device="cpu")
```

Use `model.save()` / `LanguageModel.load()` for model-only checkpoints and
`trainer.save_checkpoint()` / `trainer.load_checkpoint()` to resume an
optimizer and trainer state.

## Generation

Generation encodes the source once and can reuse decoder attention state.
The returned tensor contains generated token IDs only; it does not prepend the
source or a beginning-of-sequence token.

```python
model.eval()
generated_ids = model.generate(
    input_ids=source_ids,
    max_new_tokens=64,
    temperature=0.0,  # greedy decoding
    use_cache=True,
)
```

Set `temperature` above zero for sampling. `top_k` optionally limits sampling
to the highest-scoring tokens.

## Compilation

Use `wind.compile()` to compile a model through PyTorch. Validate public input
contracts before entering a compiled hot loop, then warm up with a
representative batch.

```python
import wind

model.validate_inputs(input_ids=source_ids, labels=labels)
compiled = wind.compile(
    model,
    mode="reduce-overhead",
    fullgraph=False,
    warmup=True,
    warmup_input={"input_ids": source_ids, "labels": labels},
)
output = compiled(input_ids=source_ids, labels=labels)
```

Compilation is workload- and environment-dependent. Benchmark eager and
compiled execution with the same shapes, dtype, device, warmup policy, and
timing scope. Avoid assuming that a compiled wrapper alone demonstrates a
speedup.

## Parameter-efficient adaptation

WiND includes a region-aware adapter system for compatible linear,
embedding, and parameter targets. It supports low-rank and weight-decomposed
adapters, semantic target selection, adapter-only checkpoints, merging, and
training only the intended parameters.

```python
from wind import (
    LoRAConfig,
    discover_lora_targets,
    inject_lora,
    lora_parameter_report,
    mark_only_lora_trainable,
)

targets = discover_lora_targets(model)
adapted = inject_lora(model, LoRAConfig(
    rank=8,
    alpha=16,
    dropout=0.05,
    regions=("encoder", "depth", "decoder", "lm_head"),
))
mark_only_lora_trainable(model)
print(lora_parameter_report(model))
```

Valid semantic regions include `wide`, `pkm`, `encoder`, `compressor`,
`bank`, `depth`, `alpha`, `decoder`, `lm_head`, `embeddings`, and `custom`.
Use `include` and `exclude` patterns for exact nested target control. See
`examples/lora_wind25m.py` for a complete fine-tuning example.

## Architecture-level inspection

[WiNDBreaker](./windbreaker/) is a separate library built on top of WiND for
non-invasive architecture-level debugging. It uses PyTorch forward hooks and
can report executed reasoning layers, memory-read events, state metadata, and
gradient-flow presence without modifying the model.

```python
from windbreaker import inspect_model

with inspect_model(model) as inspection:
    output = model(input_ids=source_ids, labels=labels)
    output.loss.backward()

for event in inspection.report.layer_applications:
    print(event.reasoning_pass, event.module, event.state.shape, event.gradient_seen)
```

The default report captures tensor metadata only. Use
`state_stats="summary"` only when reductions such as mean and standard
deviation are needed, because they add measurement overhead.

## Repository layout

| Path | Purpose |
| --- | --- |
| `wind/` | public model, training, adapter, and profiling APIs |
| `windbreaker/` | optional architecture-level inspection package |
| `examples/` | runnable usage examples |
| `benchmarks/` | repeatable performance benchmarks |
| `experiments/` | controlled architecture and performance experiments |
| `tests/` | correctness and regression tests |

## Testing

```bash
python -m pytest -q
```

The test suite includes CPU coverage. Hardware-specific performance and
accelerator checks depend on the local PyTorch and device installation.