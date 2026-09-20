# WiND 2 API Reference

WiND (WideNDepth) is a PyTorch framework for WideNDepth-style language models with FlashPKM and LoRA support. Two-layer architecture: **WiNC** (low-level backend) and **WiND** (high-level frontend).

```python
import wind
from wind.language import LMConfig, LanguageModel, LMTrainer
from wind import Config, build, Trainer
```

---

## Core API: LanguageModel

### `LanguageModel`

Prefix-LM with separate source encoding and target decoding paths.

```python
from wind.language import LanguageModel, LMConfig

config = LMConfig(vocab_size=256, dim=128, heads=4, wide_type="pkm")
model = LanguageModel(config)

# Training
output = model(input_ids, labels=targets)
output.loss.backward()

# Inference
generated = model.generate(input_ids, max_new_tokens=32, temperature=0.7)

# Save/Load
model.save("model.pt")
model = LanguageModel.load("model.pt")
```

**Methods:** `forward()`, `encode()`, `generate()`, `save()`, `load()`, `validate_inputs()`

### `LMConfig`

Frozen dataclass for model configuration:

| Field | Default | Description |
|-------|---------|-------------|
| `vocab_size` | — *(required)* | Vocabulary size |
| `dim` | `256` | Model dimension |
| `heads` | `8` | Attention heads |
| `width` | `2` | Wide branch count |
| `encoder_depth` | `2` | Encoder layers |
| `depth` | `4` | Depth layers |
| `iterations` | `1` | Reasoning iterations |
| `decoder_depth` | `2` | Decoder layers |
| `state_tokens` | `16` | Reasoning state tokens |
| `bank_tokens` | `64` | Feature bank size |
| `max_source_length` | `512` | Max input length |
| `max_target_length` | `512` | Max target length |
| `mlp_ratio` | `4.0` | FFN expansion |
| `dropout` | `0.0` | Dropout rate |
| `pad_token_id` | `0` | PAD token (distinct from BOS/EOS) |
| `bos_token_id` | `1` | BOS token |
| `eos_token_id` | `2` | EOS token |
| `norm` | `"rmsnorm"` | `"rmsnorm"` or `"layernorm"` |
| `use_rope` | `True` | Rotary position embeddings |
| `rope_theta` | `10000.0` | RoPE frequency |
| `rope_max_seq_len` | `2048` | Max RoPE length |
| `wide_type` | `"dense"` | `"dense"` or `"pkm"` |
| `encoder_type` | `"standard"` | `"standard"` or `"small"` |
| `use_alpha_learning` | `False` | Learnable state interpolation |
| `alpha_init` | `0.0` | Initial alpha value |
| `checkpointing` | `False` | Gradient checkpointing |
| `pkm_memory_size` | `4096` | PKM slots per factor |
| `pkm_num_factors` | `2` | Factorized sub-key tables |
| `pkm_heads` | `4` | Retrieval heads |
| `pkm_topk` | `32` | Global top-k per head |
| `pkm_topk_per_factor` | `None` | Per-factor top-k |
| `pkm_key_dtype` | `"float32"` | `"float32"\|"float16"\|"bfloat16"\|"int8"` |
| `pkm_value_dtype` | `"float32"` | `"float32"\|"float16"\|"bfloat16"` |
| `pkm_similarity` | `"cosine"` | `"cosine"` or `"dot"` |
| `pkm_exact_candidate_pruning` | `False` | Prune Cartesian candidates |

### `LMOutput`

```python
@dataclass
class LMOutput:
    logits: torch.Tensor      # [batch, seq, vocab_size]
    loss: torch.Tensor | None # Scalar (training mode)
    token_count: torch.Tensor # 0-d int tensor (valid target tokens)
```

### `ReasoningState`

```python
@dataclass
class ReasoningState:
    tokens: torch.Tensor  # [batch, state_tokens, dim]
    # Methods: to(), detach()
```

---

## High-Level API

### `Config`

Compact configuration for `build()`:

```python
from wind import Config

config = Config(
    dim=256,
    heads=8,
    depth=6,
    wide_type="dense",  # or "pkm"
    attention="mla",    # "mla" or "nsa"
    pkm_similarity="cosine",
)

# Immutable update
smaller = config.evolve(depth=4)
```

### `build()`

```python
from wind import build

model = build(config)
# Or with custom components:
model = build(config, wide=my_wide, depth=my_depth_layers)
```

### `ModelBuilder`

Fluent builder pattern:

```python
from wind import ModelBuilder

model = (ModelBuilder
    .from_config(Config(dim=128))
    .with_wide(custom_wide)
    .with_encoder(custom_encoder)
    .build())
```

### `Trainer`

Foundation training loop for tensor models:

```python
from wind import Trainer

trainer = Trainer(
    model,
    loss=loss_fn,          # fn(output, target) -> loss
    optimizer=None,        # Defaults to fused AdamW
    device=None,           # Auto
    amp=True,              # bf16 mixed precision
    grad_accumulation=4,
    clip_grad=1.0,
)

metrics = trainer.train_epoch(batches)
metrics = trainer.evaluate(batches)
# TrainingMetrics: avg_loss, total_loss, samples, steps
```

### `RegularizedTrainer`

```python
from wind import RegularizedTrainer, Orthogonality, DiverseBank

trainer = RegularizedTrainer(
    model,
    loss=loss_fn,
    regularizers=[
        Orthogonality(0.1),   # Branch orthogonality
        DiverseBank(0.05),    # Bank diversity
    ],
)
```

### `TrainingContext`

Checkpoint/logging wrapper:

```python
from wind import TrainingContext

ctx = TrainingContext(trainer, checkpoint_dir="ckpts/")
ctx.on_optimizer_step(lambda info: print(f"Step {info.global_step}"))
ctx.save_checkpoint("ckpt.pt")
```

### `LMTrainer`

Language-model-specific trainer:

```python
from wind.language import LMTrainer

trainer = LMTrainer(
    model,
    lr=3e-4,
    weight_decay=0.01,
    device="cuda",
    amp=True,
    amp_dtype=torch.bfloat16,
    grad_accumulation=4,
    clip_grad=1.0,
    logger="console",  # or "json" or callable
    on_optimizer_step=callback,
    on_step_end=callback,
)

loss = trainer.step(dataloader)       # One epoch
losses = trainer.fit(dataloader, epochs=3)
result = trainer.evaluate(dataloader)
trainer.save_checkpoint(path)
trainer.load_checkpoint(path)
```

**Callbacks:** `on_step_end(TrainerState)`, `on_optimizer_step(OptimizerStepInfo)`

---

## LoRA Adapters

### `LoRAConfig`

```python
from wind import LoRAConfig

policy = LoRAConfig(
    rank=8,                    # Adapter rank
    alpha=16,                  # Scaling (default: rank)
    dropout=0.05,              # Adapter dropout
    initialization="lora",    # "lora"|"gaussian"|"zeros"
    enabled=True,
    bias="none",              # "none"|"all"|"adapted"
    regions=("pkm", "bank", "depth"),  # Target regions
    include=("encoder.*",),   # Fnmatch include patterns
    exclude=("lm_head",),     # Fnmatch exclude patterns
    adapt_parameters=True,    # Adapt scalar/matrix params
    freeze_base=True,         # Freeze base model
)
```

**Target Regions:**

| Region | Targets |
|--------|---------|
| `wide` | Dense Wide branch layers |
| `pkm` | PKM memory keys/values |
| `encoder` | Encoder transformer blocks |
| `compressor` | Compression layer |
| `bank` | Feature bank |
| `depth` | Depth/reasoning blocks |
| `alpha` | Alpha interpolation parameter |
| `decoder` | Decoder blocks |
| `lm_head` | Output projection |
| `embeddings` | Token/position embeddings |
| `custom` | All other parameters |

### LoRA Functions

```python
from wind import (
    inject_lora, remove_lora, merge_lora, unmerge_lora,
    save_lora_adapters, load_lora_adapters,
    lora_parameters, lora_parameter_report,
    discover_lora_targets, mark_only_lora_trainable,
)

applied = inject_lora(model, policy)
print(lora_parameter_report(model))
save_lora_adapters(model, "adapter.pt", policy)
load_lora_adapters(model, "adapter.pt")
merge_lora(model)      # Merge for inference
unmerge_lora(model)    # Unmerge for further training
remove_lora(model, merge=True)  # Remove, optionally merge first
```

---

## Tokenizer & Data

### `ByteTokenizer`

```python
from wind import ByteTokenizer

tokenizer = ByteTokenizer(pad_token_id=0, bos_token_id=1, eos_token_id=2)
ids = tokenizer.encode("hello world")  # torch.Tensor
text = tokenizer.decode(ids)           # str
```

### `TextCollator`

```python
from wind import TextCollator

collator = TextCollator(pad_token_id=0)
batch = collator([{"input_ids": ids1, "labels": labels1}, ...])
```

### `Sequential`

```python
from wind import Sequential

# Accepts positional args or a single list/tuple
model = Sequential(layer1, layer2, layer3)
model = Sequential([layer1, layer2])
```

---

## Profiling

### `profile()` Context Manager

```python
from wind import profile

with wind.profile(
    warmup_steps=3,
    active_steps=10,
    repeat=1,
    record_shapes=True,
    with_flops=True,
    arch_profile=False,     # True for per-layer analysis
    json_output=False,
    trace_dir=None,
) as prof:
    result = model(inputs)
    loss.backward()
    prof.step()

prof.print()      # Console summary
prof.save("out.json")  # Save results
```

**ProfileConfig fields:** `warmup_steps`, `active_steps`, `repeat`, `record_shapes`, `record_modules`, `with_stack`, `with_flops`, `arch_profile`, `trace_dir`, `json_output`, `suppress_warnings`

### Architecture Profiling

```python
from wind import ArchitectureProfiler

prof = ArchitectureProfiler(model)
prof.enable()
try:
    result = model(inputs)
    loss.backward()
finally:
    summary = prof.disable()

print(summary.tree())   # Hierarchical view
print(summary.table())  # Flat per-layer table
```

### Memory Profiling

```python
from wind import profile_memory, format_memory_report

with wind.profile() as prof:
    result = model(inputs)
wind.profile_memory(50)  # Sample for 50ms
print(wind.format_memory_report())
```

### Bottleneck Analysis

```python
with wind.profile(arch_profile=True) as prof:
    result = model(inputs)
    prof.step()

for b in prof.bottlenecks(top_n=5):
    print(f"{b.layer_name}: {b.percentage:.1f}%")
    print(f"  → {b.recommendation}")
```

### Custom Metrics & Hooks

```python
from wind import metric, register_hook

@metric("param_count")
def count_params(prof):
    return sum(p.numel() for p in model.parameters())

@register_hook
def on_start(prof, action):
    if action == "start":
        print("Profiling started")
```

**Helper functions:** `active()`, `current()`, `Summary.table()`, `Summary.to_json()`

---

## Runtime & Compilation

### `compile()` / `CompileConfig`

```python
from wind import compile, CompileConfig

compiled_model = wind.compile(
    model,
    CompileConfig(
        mode="default",         # "default"|"reduce-overhead"|"max-autotune"
        fullgraph=False,
        backend="inductor",
        cudagraphs=True,
    ),
)

# Stats tracking
wind.track_compile_start(mode="reduce-overhead")
# ... model runs ...
wind.track_compile_end()
stats = wind.get_compile_stats()
wind.reset_compile_stats()
```

### `HardwareProfile` / `OptimizedBackend`

```python
from wind import HardwareProfile, OptimizedBackend

profile = HardwareProfile(
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
)
backend = OptimizedBackend(profile)
optimized_model = backend.prepare(model)
optimizer = backend.optimizer(model.parameters(), lr=3e-4)
```

---

## Error Handling & Assertions

### Guard Functions

```python
from winc import (
    WindError, WindWarning, WindDeprecationWarning, WindAssertError,
    guard, set_guard_enabled, set_guard_strict,
)

winc.set_guard_enabled(False)  # Bypass assertions
winc.set_guard_strict(True)    # Warnings become errors

guard("info", "message", "context")
guard("warning", "msg", "context")
guard("error", "msg", "context", exc=WindError)
```

### Assertion Functions

```python
from winc import (
    assert_tensor, assert_finite, assert_dtype, assert_device,
    assert_shape, assert_shape_compatible,
    warn_sync, warn_dtype_mismatch, warn_device_mismatch,
)

assert_tensor(x, "MyModule.input")
assert_finite(loss, "loss", module)
assert_dtype(ids, torch.long, "input_ids")
assert_shape(x, (batch, seq, dim), "hidden_states")
```

---

## Logging

```python
from wind import Logger, console_logger, json_logger, make_logger

logger = make_logger("console")  # Built-in console logger
logger = make_logger("json")     # JSON lines logger
logger = make_logger(my_callable_fn)

logger({"step": 1, "loss": 0.5})
```

---

## Feature Bank & Caching

### `GenerationCache`

```python
from wind import GenerationCache

cache = GenerationCache()
cache.set_cross_cache(key_tensor, value_tensor)
```

### `LayerCache`, `CacheView`

```python
from wind import LayerCache, CacheView

cache = LayerCache(max_seq_len=2048, num_layers=12)
view = cache.get_view(layer_idx=0)
# view stores/retrieves KV cache for a specific layer
```

---

## PKM (FlashPKM)

```python
from wind import FactorizedPKM, PKMWide, QueryEncoder, pkm_available

if pkm_available():
    # Direct PKM instantiation
    pkm = FactorizedPKM(
        query_dim=128,
        memory_size=4096,
        value_dim=256,
        num_factors=2,
        heads=4,
        topk=32,
        similarity="cosine",
        key_dtype=torch.float32,
    )
    output = pkm(query_tensor)  # [B, S, H, F*D] -> [B, S, V]

    # Or use PKMWide directly
    wide = PKMWide(
        dim=256,
        memory_size=4096,
        query_dim=128,
        num_factors=2,
        heads=4,
        output_mode="gated",  # "none"|"residual"|"gated"
    )
    pkm.prepare_for_inference(dtype=torch.float16)
```

---

## Complete Training Example

```python
import torch
from wind.language import LMConfig, LanguageModel, LMTrainer
from wind import ByteTokenizer

# Setup
tokenizer = ByteTokenizer()
config = LMConfig(
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    dim=128, heads=4,
    wide_type="pkm",
    pkm_memory_size=3900,
    use_alpha_learning=True,
    max_source_length=128,
    max_target_length=128,
)

model = LanguageModel(config)
trainer = LMTrainer(
    model, lr=3e-4,
    weight_decay=0.01,
    device="cuda",
    amp=True,
    grad_accumulation=4,
    clip_grad=1.0,
    logger="console",
)

# Training loop
for epoch in range(3):
    loss = trainer.step(dataloader)
    print(f"Epoch {epoch+1}: loss={loss}")

model.save("model.pt")
```

---

## Module Summary

### From `wind` (WiND frontend)
`Sequential`, `ByteTokenizer`, `TextCollator`, `Config`, `build`, `configure`, `ModelBuilder`, `Trainer`, `Regularizer`, `Orthogonality`, `DiverseBank`, `RegularizedTrainer`, `TrainingContext`, `KnowledgeTrainer`, `LMConfig`, `LanguageModel`, `LMTrainer`, `LoRAConfig`, `inject_lora`, `remove_lora`, `merge_lora`, `unmerge_lora`, `save_lora_adapters`, `load_lora_adapters`, `discover_lora_targets`, `mark_only_lora_trainable`, `lora_parameters`, `lora_parameter_report`, plus full profiling/compiler/error/logging APIs.

### From `winc` (WiNC backend)
Core modules: `WindModule`, `Wide`, `WideStack`, `Depth`, `RMSNorm`, `FeedForward`, `SwiGLU`. Attention: `RotaryEmbedding`/`RoPE`, `MLA`, `NSA`, `Attention`. Architecture: `LearnedQueryCompressor`, `Compressor`, `FeatureBank`, `AdaptiveFeatureBank`, `Retrieval`, `ReasoningDepth`, `WideNDepth`, `EncoderDecoder`. Adapters, losses, runtime, logging, caching, and FlashPKM bridge.
```