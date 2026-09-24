# WiND

## CPU quick start

This is a dependency-free CPU example: it constructs a generic WiND model,
runs a forward pass, runs backward, and prints the resulting state shape.  It
does not download data or require CUDA or FlashPKM.

```python
import torch
import wind

model = wind.build(wind.Config(
    dim=16, heads=4, depth=2, iterations=2,
    state_tokens=4, bank_tokens=6, read_tokens=2,
))
x = torch.randn(2, 8, 16)
y = model(x)
y.square().mean().backward()
print(y.shape)  # torch.Size([2, 4, 16])
```

Install the dense path with `pip install .`.  PKM is optional: `pip install
"wind[pkm]"`.  A dense model does not import or require FlashPKM.

## Generic build() vs LanguageModel: execution contract

Both APIs are supported.  They intentionally use different reasoning-loop
semantics, so `depth` and `iterations` must not be treated as interchangeable.

Definitions used throughout WiND:

- **depth**: the number of distinct Depth (reasoning) layers.
- **iterations**: the number of times the model passes its state back through
  the Depth stack while reading its Feature Bank.  It is a reasoning-pass count,
  not a layer count.

| Path | `depth` | What one `iteration` executes | Worked `depth=6, iterations=1` example | Layer applications |
| --- | --- | --- | --- | --- |
| `wind.build(Config(...))` | Number of layers in `ReasoningDepth.layers` | One layer selected cyclically: `layers[iteration % depth]`, with one bounded retrieval | Executes `layers[0]` once; layers 1–5 do not execute | exactly `iterations` |
| `LanguageModel(LMConfig(...))` | Number of modules in `model.depth` | The complete stack, in order, with cross-attention to the bank per layer | Executes depth layers 0, 1, 2, 3, 4, 5 once | exactly `depth * iterations` |

For generic build, `depth=6, iterations=8` executes layers `0,1,2,3,4,5,0,1`.
For LanguageModel, `depth=6, iterations=8` executes all six layers on each of
eight reasoning passes (48 layer applications).  This is existing behavior;
WiND documents it rather than silently redesigning either API.

### State and Bank token budgets

Both paths expose independent budgets:

- `state_tokens`: Compressor output tokens, i.e. the state consumed by Depth.
- `bank_tokens`: Feature Bank capacity, i.e. memory available to retrieval.

`Config.state_tokens` defaults to 64, matching its existing `bank_tokens=64`
default for compatibility.  Set both explicitly whenever their difference is
architecturally important.  `LMConfig` has always exposed both independently.

### Bank behavior

“Read-only” alone is ambiguous.  The banks have these independent properties:

| Path | Immutable during recurrence? | Lifetime | Autograd behavior |
| --- | --- | --- | --- |
| Generic `FeatureBank` (default) | Yes. A local tensor is constructed once; Depth only reads/projections it. | Request-local; no module cache persists across forwards. | Detached from encoder/task autograd (`detach=True` default); bank normalizer also runs under `no_grad`. Set `detach=False` for differentiable end-to-end bank construction. |
| Generic `AdaptiveFeatureBank` | Yes; selection is made once before recurrence. | Request-local; no persistent cache. | Same `detach` policy. With `detach=False`, selected score values softmax-weight selected entries so `score.weight` receives a task-loss gradient. Top-k membership itself remains discrete. |
| `LanguageModel.bank` | Yes. `LearnedQueryCompressor` creates bank tokens once, then every depth layer cross-attends to the same tensor. | Request-local (`encode()` result); no hidden persistent bank. | Differentiable end-to-end. Its learned queries, attention, and normalization receive task gradients. |

## WiNDBreaker inspection

WiNDBreaker is a separate package (`WiNDBreaker/`) that depends on `wind`, not
the reverse.  It uses ordinary PyTorch forward hooks and never monkeypatches
the model.  The hook maps generic cyclic iterations and LanguageModel full-stack
iterations to a uniform report.

```python
from windbreaker import inspect_model

with inspect_model(model) as inspection:
    y = model(x)
    y.square().mean().backward()

for event in inspection.report.layer_applications:
    print(event.reasoning_pass, event.module, event.state.shape, event.gradient_seen)
```

The default state statistics are metadata only (shape, dtype, device, numel,
and `requires_grad`), avoiding synchronization and tensor reductions.  Pass
`state_stats="summary"` only when mean/std measurements are worth their extra
work.  Detach handles with `inspection.detach()` or exit the context manager.
