"""LanguageModel: prefix-LM with WideNDepth internals + decoder.

Only source tokens enter the knowledge/reasoning path. Target tokens enter a
causal decoder after a right shift, preventing teacher-forcing answer leakage.
The encoder is bidirectional; this is a conditional/prefix LM, not a decoder-only
LM trained by passing the same full sequence to both source and target.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from torch.profiler import record_function
from torch._dynamo import allow_in_graph

from winc.attention import Attention
from winc.architecture import LearnedQueryCompressor
from winc.cache import GenerationCache
from winc.modules import FeedForward, RMSNorm, Wide, WindModule
from winc._internal.guard import (
    assert_finite,
    assert_tensor,
    assert_dtype,
    assert_device,
    assert_shape,
    warn_sync,
    GUARD_ENABLED,
)
from .config import LMConfig, _map_pkm_dtype, _validate_pkm_config
from .checkpoint import load_checkpoint, save_checkpoint
from winc.pkm import make_pkm_wide

allow_in_graph(record_function)


def _profile_range(name: str):
    """Emit eager profiler labels without introducing a Dynamo graph break."""
    if torch.compiler.is_compiling() or not torch.autograd._profiler_enabled():
        return nullcontext()
    return record_function(name)


@dataclass
class ReasoningState:
    """Reusable normalized latent memory [batch, state_tokens, dim].

    Request-local and differentiable, with no hidden mutable state on the model.
    Image/video/audio feature encoders can produce the same interface.
    """
    tokens: torch.Tensor

    def to(self, *args, **kwargs):
        return ReasoningState(self.tokens.to(*args, **kwargs))

    def detach(self):
        return ReasoningState(self.tokens.detach())


@dataclass
class LMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    token_count: torch.Tensor = None  # 0-d int tensor; stays on-device under torch.compile


class _Block(nn.Module):
    def __init__(self, config, *, cross=False):
        super().__init__()
        norm = RMSNorm if config.norm == "rmsnorm" else nn.LayerNorm
        self.norm1 = norm(config.dim)
        self.norm2 = norm(config.dim)
        self.attn = Attention(config.dim, config.heads, config.dropout,
                              config.use_rope, config.rope_theta, config.rope_max_seq_len)
        self.ffn = FeedForward(config.dim, int(config.dim * config.mlp_ratio), config.dropout)
        self.cross = Attention(config.dim, config.heads, config.dropout,
                               config.use_rope, config.rope_theta, config.rope_max_seq_len) if cross else None
        self.cross_norm = norm(config.dim) if cross else None

    def forward(self, x, memory=None, *, mask=None, causal=False, cache=None,
                use_cache=False, position_offset=0, layer_idx=0):
        from winc.cache import GenerationCache
        using_gen_cache = isinstance(cache, GenerationCache) and use_cache
        if using_gen_cache:
            # CacheView appends self K/V in-place before attention reads the
            # complete prefix; no per-token torch.cat is required.
            old_self = cache.get_view(layer_idx, "self")
            old_cross = cache.get_view(layer_idx, "cross")
        else:
            old_self, old_cross = cache if cache is not None else (None, None)
        y, new_self = self.attn(self.norm1(x), mask=mask, causal=causal,
                                cache=old_self, use_cache=use_cache,
                                position_offset=position_offset)
        x = x + y
        new_cross = None
        if self.cross is not None:
            y, new_cross = self.cross(self.cross_norm(x), memory, cache=old_cross,
                                      static=True, use_cache=use_cache)
            x = x + y
        x = x + self.ffn(self.norm2(x))
        if using_gen_cache:
            return x, None
        return x, (new_self, new_cross) if use_cache else None


class LanguageModel(WindModule):
    """Text IDs/features -> Wide -> Encoder -> bank + Depth -> token decoder.

    ``forward(input_ids, labels=continuation_ids)`` shifts labels internally.
    ``encode`` returns reusable states; ``generate`` returns only new token IDs,
    including EOS and subsequent PAD, without a prepended BOS or source prompt.
    """
    def __init__(self, config: LMConfig | None = None, **kwargs):
        super().__init__()
        if config is not None and kwargs:
            raise TypeError("pass config or keyword fields, not both")
        self.config = config if config is not None else LMConfig(**kwargs)
        if not isinstance(self.config, LMConfig):
            raise TypeError("config must be LMConfig")
        c = self.config

        # Allow custom module injection via config.custom_modules dict.
        # Keys are attribute names; values are nn.Module instances or factory callables.
        custom = c.custom_modules.get("norm", None)
        norm_cls = custom if custom is not None else (RMSNorm if c.norm == "rmsnorm" else nn.LayerNorm)

        self.embedding = nn.Embedding(c.vocab_size, c.dim, padding_idx=c.pad_token_id)
        self.source_position = nn.Embedding(c.max_source_length, c.dim)
        self.target_position = nn.Embedding(c.max_target_length, c.dim)

        # --- Wide stage: dense or PKM ---
        custom_wide = c.custom_modules.get("wide", None)
        if custom_wide is not None:
            self.wide = custom_wide
        elif c.wide_type == "pkm":
            _validate_pkm_config(c)
            self.wide = make_pkm_wide(
                dim=c.dim,
                memory_size=c.pkm_memory_size,
                query_dim=c.dim // 2,
                value_dim=c.dim,
                num_factors=c.pkm_num_factors,
                heads=c.pkm_heads,
                topk=c.pkm_topk,
                topk_per_factor=c.pkm_topk_per_factor,
                similarity=c.pkm_similarity,
                output_mode="none",
                key_dtype=_map_pkm_dtype(c.pkm_key_dtype),
                value_dtype=_map_pkm_dtype(c.pkm_value_dtype),
                exact_candidate_pruning=c.pkm_exact_candidate_pruning,
            )
        else:
            self.wide = Wide(*[nn.Sequential(norm_cls(c.dim), FeedForward(
                c.dim, int(c.dim * c.mlp_ratio), c.dropout)) for _ in range(c.width)])

        # --- Encoder: standard or small ---
        custom_encoder = c.custom_modules.get("encoder_block", None)
        block_cls = custom_encoder if custom_encoder is not None else _Block
        enc_depth = c.encoder_depth
        if c.encoder_type == "small":
            enc_depth = max(1, enc_depth // 2)
        self.encoder = nn.ModuleList([block_cls(c) for _ in range(enc_depth)])

        self.compressor = LearnedQueryCompressor(
            c.dim, c.state_tokens, heads=c.heads, dropout=c.dropout,
            use_rope=c.use_rope, rope_theta=c.rope_theta,
            rope_max_seq_len=c.rope_max_seq_len, norm=c.norm,
        )
        self.bank = LearnedQueryCompressor(
            c.dim, c.bank_tokens, heads=c.heads, dropout=c.dropout,
            use_rope=c.use_rope, rope_theta=c.rope_theta,
            rope_max_seq_len=c.rope_max_seq_len, norm=c.norm,
        )

        # --- Depth/Reasoning ---
        custom_depth = c.custom_modules.get("depth_block", None)
        depth_block_cls = custom_depth if custom_depth is not None else _Block
        self.depth = nn.ModuleList([depth_block_cls(c, cross=True) for _ in range(c.depth)])

        self.state_norm = norm_cls(c.dim)
        self.decoder = nn.ModuleList([_Block(c, cross=True) for _ in range(c.decoder_depth)])
        self.final_norm = norm_cls(c.dim)
        self.lm_head = nn.Linear(c.dim, c.vocab_size, bias=False)

        # --- Alpha learning: learnable interpolation between encoder output and state ---
        if c.use_alpha_learning:
            self.alpha = nn.Parameter(torch.tensor(c.alpha_init))
        else:
            self.alpha = None

    def _ids(self, ids, name, limit):
        if ids.ndim != 2 or ids.size(0) < 1 or not 0 < ids.size(1) <= limit:
            raise ValueError(f"{name} must be nonempty [batch, tokens], length <= {limit}")
        if ids.dtype != torch.long:
            raise ValueError(f"{name} must have dtype torch.long")
        # Content validation is valuable at the public eager boundary, but a
        # tensor-valued assertion is data dependent and cannot be guarded by
        # Dynamo in a fullgraph region.  Keep the eager contract and let a
        # compiled caller explicitly preflight with ``validate_inputs`` before
        # entering its hot loop.  Valid IDs are still required by embedding.
        if not torch.compiler.is_compiling():
            out_of_range = ((ids < 0) | (ids >= self.config.vocab_size)).any()
            if bool(out_of_range):
                raise ValueError(f"{name} contains IDs outside the vocabulary")

    def validate_inputs(self, input_ids=None, *, labels=None,
                        decoder_input_ids=None, attention_mask=None) -> None:
        """Eager preflight validation for a subsequent compiled hot loop.

        This method intentionally performs the content checks excluded from a
        Dynamo fullgraph. It does not allocate model state or mutate a cache.
        """
        c = self.config
        if input_ids is not None:
            self._ids(input_ids, "input_ids", c.max_source_length)
        if labels is not None:
            clean = labels.masked_fill(labels == -100, c.pad_token_id)
            self._ids(clean, "labels", c.max_target_length)
        if decoder_input_ids is not None:
            self._ids(decoder_input_ids, "decoder_input_ids", c.max_target_length)
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError("attention_mask must have source shape and contain only 0/1")
            if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
                raise ValueError("attention_mask must contain only 0/1")

    def _run(self, block, x, memory=None, **kwargs):
        if self.config.checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(lambda a, b: block(a, b, **kwargs)[0], x, memory,
                              use_reentrant=False)
        return block(x, memory, **kwargs)[0]

    def encode(self, input_ids=None, *, attention_mask=None, features=None):
        """Encode source IDs OR adapter features; True/1 mask entries are valid.

        Features must be [batch, tokens, dim] on the model device. No target
        continuation may be present in this source during prefix-LM training.
        """
        if (input_ids is None) == (features is None):
            raise ValueError("provide exactly one of input_ids or features")
        c = self.config
        if input_ids is not None:
            self._ids(input_ids, "input_ids", c.max_source_length)
            x = self.embedding(input_ids)
            valid = input_ids != c.pad_token_id if attention_mask is None else attention_mask
        else:
            if (features.ndim != 3 or features.size(-1) != c.dim or
                    features.size(0) < 1 or not 0 < features.size(1) <= c.max_source_length):
                raise ValueError("features must be nonempty [batch, tokens, dim] within source limit")
            x = features
            valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if attention_mask is None else attention_mask
        if valid.shape != x.shape[:2]:
            raise ValueError("attention_mask must have source shape and contain only 0/1")
        # Dynamo-safe: use torch._assert for traceable assertions that work
        # under torch.compile without a graph break.
        valid_range = (valid == 0) | (valid == 1)
        if not torch.compiler.is_compiling():
            if not bool(valid_range.all()):
                raise ValueError("attention_mask must contain only 0/1")
        valid = valid.to(device=x.device, dtype=torch.bool)
        # Each source needs at least one valid token.
        if not torch.compiler.is_compiling():
            if not bool(valid.any(dim=1).all()):
                raise ValueError("each source needs at least one valid token")
        # Positions count valid tokens: left/right padding do not alter states.
        positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
        x = x + self.source_position(positions)
        # Wide-stage residual: scale depends on which component self.wide is.
        # - Wide (sum of N branches): divide by sqrt(N) for variance stability.
        # - PKMWide (single retrieval path, output_mode="none"): no scaling needed.
        wide_out = self.wide(x)
        if hasattr(self.wide, "mode") and self.wide.mode == "sum":
            x = x + wide_out / math.sqrt(c.width)
        elif hasattr(self.wide, "output_mode") and self.wide.output_mode == "none":
            x = x + wide_out
        else:
            x = x + wide_out / math.sqrt(c.width)
        mask = valid[:, None, None, :]
        with _profile_range("encoder_stack"):
            for block in self.encoder:
                x = self._run(block, x, mask=mask)
        states, bank = self.compressor(x, mask), self.bank(x, mask)
        # Every depth layer is used on EVERY reasoning iteration.
        with _profile_range("depth_stack"):
            for _ in range(c.iterations):
                for block in self.depth:
                    states = self._run(block, states, bank)
        # Alpha learning: learnable interpolation between encoder output and state.
        if self.alpha is not None:
            with _profile_range("alpha_blend"):
                alpha = torch.sigmoid(self.alpha)
                states = alpha * states + (1 - alpha) * self.state_norm(states)
        return ReasoningState(self.state_norm(states))

    def _decode(self, ids, state, *, cache=None, use_cache=False, offset=0):
        if (state.tokens.ndim != 3 or state.tokens.size(0) != ids.size(0) or
                state.tokens.size(1) < 1 or state.tokens.size(-1) != self.config.dim):
            raise ValueError("state must have matching batch, nonempty tokens and model dim")
        if offset + ids.size(1) > self.config.max_target_length:
            raise ValueError("decoder position exceeds max_target_length")
        positions = torch.arange(offset, offset + ids.size(1), device=ids.device)
        x = self.embedding(ids) + self.target_position(positions)

        next_cache = [] if use_cache and not isinstance(cache, GenerationCache) else None
        with _profile_range("decoder_stack"):
            for i, block in enumerate(self.decoder):
                if use_cache:
                    if isinstance(cache, GenerationCache):
                        # With GenerationCache, block receives the cache directly
                        x, kv = block(x, state.tokens, causal=True,
                                      cache=cache, use_cache=True, layer_idx=i,
                                      position_offset=offset)
                    else:
                        # Legacy tuple cache path
                        x, kv = block(x, state.tokens, causal=True,
                                      cache=None if cache is None else cache[i], use_cache=True)
                    if not isinstance(cache, GenerationCache):
                        next_cache.append(kv)
                else:
                    x = self._run(block, x, state.tokens, causal=True)
        return self.lm_head(self.final_norm(x)), next_cache if use_cache and not isinstance(cache, GenerationCache) else None

    def forward(self, input_ids=None, *, labels=None, decoder_input_ids=None,
                attention_mask=None, features=None, state=None):
        if state is not None and any(v is not None for v in (input_ids, features, attention_mask)):
            raise ValueError("provide state or source inputs, not both")
        if labels is not None:
            if labels.ndim != 2 or labels.dtype != torch.long:
                raise ValueError("labels must be [batch, tokens] torch.long")
            clean = labels.masked_fill(labels == -100, self.config.pad_token_id)
            self._ids(clean, "labels", self.config.max_target_length)
            if decoder_input_ids is None:
                decoder_input_ids = torch.full_like(clean, self.config.bos_token_id)
                decoder_input_ids[:, 1:] = clean[:, :-1]
        if decoder_input_ids is None:
            raise ValueError("provide labels or decoder_input_ids")
        self._ids(decoder_input_ids, "decoder_input_ids", self.config.max_target_length)
        if state is None:
            state = self.encode(input_ids, attention_mask=attention_mask, features=features)
        with _profile_range("output_projection_and_loss"):
            logits, _ = self._decode(decoder_input_ids, state)
        loss, count = None, 0
        if labels is not None:
            # These diagnostics construct formatted Python strings on a failed
            # check. Shape/dtype have already been checked at the public
            # boundary above; omit redundant guard machinery from fullgraph.
            if not torch.compiler.is_compiling():
                assert_shape(labels, (-1, -1), "labels")
                assert_dtype(labels, torch.long, "labels", self)
            if labels.shape != decoder_input_ids.shape:
                raise ValueError("labels and decoder_input_ids must have the same shape")
            targets = labels.masked_fill(labels == self.config.pad_token_id, -100)
            count_tensor = (targets != -100).sum()
            # Compute loss in FP32 for numerical stability; BF16 has only 7 exponent
            # bits, which can cause overflow in the log-softmax of large vocabularies.
            # FP16 native cross_entropy is stable enough.
            with _profile_range("loss"):
                if logits.dtype == torch.bfloat16:
                    loss = F.cross_entropy(
                        logits.float().reshape(-1, logits.size(-1)),
                        targets.reshape(-1), ignore_index=-100, reduction="sum",
                    )
                else:
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1), ignore_index=-100, reduction="sum",
                    )
            # Normalize by token count using a tensor-valued clamp; stay on-device
            # so this works under torch.compile without a graph break.
            loss = loss / count_tensor.to(loss.dtype).clamp_min(1)
            # Return the count as a detached 0-d int tensor; the trainer keeps
            # it device-side and never calls .item() on the hot path.
            count = count_tensor.detach()
            if GUARD_ENABLED and not torch.compiler.is_compiling():
                assert_finite(loss, "loss", self)
        return LMOutput(logits, loss, count)

    @torch.inference_mode()
    def generate(
        self,
        input_ids=None,
        *,
        attention_mask=None,
        features=None,
        state=None,
        max_new_tokens=64,
        temperature=0.0,
        top_k=None,
        use_cache=True,
        generator=None,
    ):
        """Greedy at temperature=0; otherwise temperature/top-k sampling.

        Encodes once; caches decoder self-attention and state cross-attention K/V.
        Restores the caller's train/eval setting. No request cache is retained.

        Optimizations:
        - preallocates output tensor instead of torch.cat each step
        - avoids per-step CPU sync from done.all().item()
        - minimizes repeated config lookups
        - avoids unnecessary logits.clone()
        - uses in-place writes where safe
        """
        if type(max_new_tokens) is not int or not 0 < max_new_tokens <= self.config.max_target_length:
            raise ValueError("max_new_tokens must be within [1, max_target_length]")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= self.config.vocab_size):
            raise ValueError("top_k must be within [1, vocab_size]")
        if state is not None and any(v is not None for v in (input_ids, features, attention_mask)):
            raise ValueError("provide state or source inputs, not both")

        was_training = self.training
        self.eval()
        try:
            state = state if state is not None else self.encode(
                input_ids,
                attention_mask=attention_mask,
                features=features,
            )

            device = state.tokens.device
            batch_size = state.tokens.size(0)

            bos_token_id = self.config.bos_token_id
            eos_token_id = self.config.eos_token_id
            pad_token_id = self.config.pad_token_id
            vocab_size = self.config.vocab_size
            heads = self.config.heads
            dim = self.config.dim

            # Preallocate generated tokens: [B, max_new_tokens]
            out = torch.full(
                (batch_size, max_new_tokens),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )

            # Current input token for cached decoding starts from BOS
            cur = torch.full(
                (batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=device,
            )

            # Tracks which sequences have already emitted EOS
            done = torch.zeros(batch_size, dtype=torch.bool, device=device)

            cache = None
            if use_cache:
                head_dim = dim // heads
                cache = GenerationCache(
                    batch_size=batch_size,
                    max_seq_len=max_new_tokens + 1,  # BOS + generated tokens
                    num_heads=heads,
                    head_dim=head_dim,
                    dtype=next(self.parameters()).dtype,
                    device=device,
                    n_layers=len(self.decoder),
                )

                # Pre-project encoder/state KV once per decoder layer
                for layer_idx, block in enumerate(self.decoder):
                    cross = getattr(block, "cross", None)
                    if cross is not None and hasattr(cross, "project_kv"):
                        k, v = cross.project_kv(state.tokens)
                        cache.set_cross_cache_for_layer(layer_idx, k, v)

            # Optional uncached history path if use_cache=False
            # We keep history only in this branch.
            if not use_cache:
                history = torch.full(
                    (batch_size, 1),
                    bos_token_id,
                    dtype=torch.long,
                    device=device,
                )

            produced = max_new_tokens

            for step in range(max_new_tokens):
                if use_cache:
                    logits, next_cache = self._decode(
                        cur,
                        state,
                        cache=cache,
                        use_cache=True,
                        offset=step,
                    )
                    if not isinstance(cache, GenerationCache):
                        cache = next_cache
                else:
                    logits, next_cache = self._decode(
                        history,
                        state,
                        cache=None,
                        use_cache=False,
                        offset=0,
                    )

                scores = logits[:, -1, :]

                # We only need a mutable tensor if we're going to modify values.
                # clone() is safer if _decode returns a view into something reused,
                # but do it once here rather than more expensive patterns.
                scores = scores.clone()

                # Never emit PAD/BOS
                scores[:, pad_token_id] = -torch.inf
                scores[:, bos_token_id] = -torch.inf

                if temperature == 0.0:
                    token = torch.argmax(scores, dim=-1)
                else:
                    if temperature != 1.0:
                        scores.div_(temperature)

                    if top_k is not None and top_k < vocab_size:
                        topk_vals = torch.topk(scores, top_k, dim=-1).values
                        cutoff = topk_vals[:, -1:]
                        scores.masked_fill_(scores < cutoff, -torch.inf)

                    probs = torch.softmax(scores, dim=-1)
                    token = torch.multinomial(probs, 1, generator=generator).squeeze(1)

                # Once a sequence is done, keep padding it
                token = torch.where(done, torch.full_like(token, pad_token_id), token)

                out[:, step] = token
                done |= token.eq(eos_token_id)

                if not use_cache:
                    history = torch.cat((history, token.unsqueeze(1)), dim=1)

                cur = token.unsqueeze(1)

                # Avoid device->host sync every step:
                # only sync every 8 steps and on the final step.
                if ((step + 1) & 7) == 0 or step + 1 == max_new_tokens:
                    if bool(done.all()):
                        produced = step + 1
                        break

            return out[:, :produced]

        finally:
            self.train(was_training)

    def save(self, path):
        """Save reconstructible architecture and weights to one .pt file."""
        config_dict = asdict(self.config)
        config_dict.pop("custom_modules", None)
        save_checkpoint(path, {"config": config_dict, "model": self.state_dict()})

    @classmethod
    def load(cls, path, *, device="cpu"):
        """Load model-only or training checkpoints for inference (eval mode)."""
        data = load_checkpoint(path)
        # Drop custom_modules from saved config — they aren't serializable.
        config_dict = {k: v for k, v in data["config"].items() if k != "custom_modules"}
        with torch.random.fork_rng(devices=[]):
            model = cls(LMConfig(**config_dict))
        model.load_state_dict(data["model"])
        return model.to(device).eval()


