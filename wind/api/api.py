"""Beginner-friendly high-level API.

The high-level surface intentionally has a small number of concepts:

``Config`` describes the family shape, ``build`` creates a model, and
``Trainer`` provides a fast default training loop. Advanced users can still
compose the core modules directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable
import inspect

import torch
import warnings
from torch import nn

from winc.architecture import Compressor, ReasoningDepth, WideNDepth
from winc.blocks import TransformerBlock
from winc.losses import diversity_loss, orthogonality_loss
from winc.modules import FeedForward, Wide, WideStack
from winc.pkm import make_pkm_wide
from winc.runtime import HardwareProfile, OptimizedBackend


@dataclass(frozen=True)
class Config:
    """Small, stable set of knobs shared by WND-family models.

    ``depth`` is the number of distinct reasoning layers.  In the generic
    :func:`build` path, ``iterations`` is the number of *layer applications*:
    :class:`winc.ReasoningDepth` selects ``layers[i % depth]`` once for each
    iteration.  This intentionally differs from ``LanguageModel``; see the
    README execution contract before changing either value.

    ``state_tokens`` controls Compressor output while ``bank_tokens`` controls
    FeatureBank capacity.  They default to the same value for compatibility,
    but are independently configurable.
    """

    dim: int
    heads: int = 8
    depth: int = 6
    width: int = 1
    iterations: int = 1
    state_tokens: int = 64
    bank_tokens: int = 64
    read_tokens: int = 8
    attention: str = "mla"
    mlp_ratio: float = 4.0
    wide_passes: int = 1
    wide_architecture: str = "ffn"
    wide_mode: str = "sum"
    wide_type: str = "dense"  # "dense" or "pkm"
    pkm_memory_size: int = 4096
    pkm_num_factors: int = 2
    pkm_heads: int = 4
    pkm_topk: int = 32
    pkm_topk_per_factor: int | None = None
    pkm_key_dtype: str = "float32"  # "float32", "float16", "bfloat16", "int8"
    pkm_value_dtype: str = "float32"  # "float32", "float16", "bfloat16"
    pkm_similarity: str = "cosine"   # "cosine" (recommended) or "dot"

    def __post_init__(self):
        if self.dim < 1 or self.heads < 1 or self.depth < 1 or self.width < 1 or self.iterations < 1:
            raise ValueError("dim, heads, depth, width and iterations must be positive")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if self.attention not in {"mla", "nsa"}:
            raise ValueError("attention must be 'mla' or 'nsa'")
        if self.wide_passes < 1:
            raise ValueError("wide_passes must be positive")
        if self.wide_architecture not in {"ffn", "transformer"}:
            raise ValueError("wide_architecture must be 'ffn' or 'transformer'")
        if self.wide_mode not in {"sum", "concat"}:
            raise ValueError("wide_mode must be 'sum' or 'concat'")
        if self.wide_type not in {"dense", "pkm"}:
            raise ValueError("wide_type must be 'dense' or 'pkm'")
        if self.pkm_key_dtype not in {"float32", "float16", "bfloat16", "int8"}:
            raise ValueError(f"pkm_key_dtype must be 'float32', 'float16', 'bfloat16', or 'int8', got: {self.pkm_key_dtype}")
        if self.pkm_value_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"pkm_value_dtype must be 'float32', 'float16', or 'bfloat16', got: {self.pkm_value_dtype}")
        if self.pkm_similarity not in {"cosine", "dot"}:
            raise ValueError("pkm_similarity must be 'cosine' or 'dot'")
        if self.state_tokens < 1 or self.bank_tokens < 1 or self.read_tokens < 1:
            raise ValueError("state_tokens, bank_tokens and read_tokens must be positive")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")

    def evolve(self, **changes: Any) -> "Config":
        """Return a new Config with specified fields replaced (immutable update)."""
        return replace(self, **changes)

    def validate(self) -> None:
        """Explicit re-validation — constructs a new Config to trigger __post_init__."""
        from dataclasses import asdict
        type(self)(**asdict(self))


def build(
    config: Config | None = None,
    *,
    wide: nn.Module | None = None,
    encoder: nn.Module | None = None,
    compressor: nn.Module | None = None,
    depth: int | nn.Module | list[nn.Module] | tuple[nn.Module, ...] | None = None,
    bank: nn.Module | None = None,
    **kwargs,
) -> WideNDepth:
    """Build a complete WND-family model from one compact configuration.

    Keyword arguments are aliases for :class:`Config` fields. The component
    arguments allow plugging in custom classes without requiring a new builder.
    """

    custom_depth = depth
    if isinstance(depth, int):
        if "depth" in kwargs:
            raise TypeError("depth was specified twice")
        kwargs["depth"] = depth
        custom_depth = None
    if config is None:
        config = Config(**kwargs)
    elif kwargs:
        raise TypeError("pass either config or keyword fields, not both")
    if not isinstance(config, Config):
        raise TypeError("config must be a winc.Config")

    return _construct_model(
        config,
        wide=wide,
        encoder=encoder,
        compressor=compressor,
        bank=bank,
        depth=custom_depth,
    )


def _construct_model(
    config: Config,
    *,
    wide: nn.Module | None = None,
    encoder: nn.Module | None = None,
    compressor: nn.Module | None = None,
    bank: nn.Module | None = None,
    depth=None,
) -> WideNDepth:
    """Internal model construction shared by build() and ModelBuilder.build()."""

    def _validate_pkm_query_dim(cfg: Config) -> None:
        query_dim = cfg.dim // 2
        head_dim = query_dim // (cfg.pkm_heads * cfg.pkm_num_factors)
        if query_dim != head_dim * cfg.pkm_heads * cfg.pkm_num_factors:
            remainder = query_dim % (cfg.pkm_heads * cfg.pkm_num_factors)
            raise ValueError(
                f"query_dim ({query_dim}) must be divisible by "
                f"pkm_heads ({cfg.pkm_heads}) * pkm_num_factors ({cfg.pkm_num_factors}). "
                f"Remainder: {remainder}. Try adjusting dim, pkm_heads, or pkm_num_factors."
            )

    def _map_dtype(s: str) -> torch.dtype:
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
        }[s]

    stages = []
    for _ in range(config.wide_passes):
        if config.wide_type == "pkm":
            _validate_pkm_query_dim(config)
            stages.append(make_pkm_wide(
                dim=config.dim,
                memory_size=config.pkm_memory_size,
                query_dim=config.dim // 2,
                value_dim=config.dim,
                num_factors=config.pkm_num_factors,
                heads=config.pkm_heads,
                topk=config.pkm_topk,
                topk_per_factor=config.pkm_topk_per_factor,
                subkey_dim=config.dim // (config.pkm_heads * config.pkm_num_factors),
                similarity=config.pkm_similarity,
                output_mode="none",
                key_dtype=config.pkm_key_dtype,
                value_dtype=config.pkm_value_dtype,
            ))
        elif config.wide_architecture == "transformer":
            branches = [TransformerBlock(config.dim, config.heads, config.mlp_ratio, attention=config.attention)
                        for _ in range(config.width)]
            stages.append(Wide(*branches, dim=config.dim, mode=config.wide_mode))
        else:
            branches = [nn.Sequential(nn.LayerNorm(config.dim), FeedForward(
                config.dim, int(config.dim * config.mlp_ratio)
            )) for _ in range(config.width)]
            stages.append(Wide(*branches, dim=config.dim, mode=config.wide_mode))

    default_wide = stages[0] if len(stages) == 1 else WideStack(*stages)
    default_encoder = TransformerBlock(
        config.dim, config.heads, config.mlp_ratio, attention=config.attention
    )
    depth_layers = [
        TransformerBlock(config.dim, config.heads, config.mlp_ratio, attention=config.attention)
        for _ in range(config.depth)
    ]

    reasoning_depth = (
        depth
        if isinstance(depth, ReasoningDepth)
        else ReasoningDepth(
            depth if depth is not None else depth_layers,
            config.dim,
            iterations=config.iterations,
            read_tokens=config.read_tokens,
        )
    )

    return WideNDepth(
        wide=wide if wide is not None else default_wide,
        encoder=encoder if encoder is not None else default_encoder,
        compressor=compressor if compressor is not None else Compressor(config.dim, config.state_tokens),
        depth=reasoning_depth,
        dim=config.dim,
        bank_tokens=config.bank_tokens,
        read_tokens=config.read_tokens,
        iterations=config.iterations,
        bank=bank,
    )


class ModelBuilder:
    """Fluent builder for fine-grained WideNDepth customization."""

    def __init__(self, config: Config):
        self.config = config
        self._wide: nn.Module | None = None
        self._encoder: nn.Module | None = None
        self._compressor: nn.Module | None = None
        self._bank: nn.Module | None = None
        self._depth: Any = None

    @staticmethod
    def from_config(config: Config) -> "ModelBuilder":
        """Initialize from a Config."""
        return ModelBuilder(config)

    def with_wide(self, module: nn.Module) -> "ModelBuilder":
        """Custom width/parallelization module."""
        self._wide = module
        return self

    def with_encoder(self, module: nn.Module) -> "ModelBuilder":
        """Custom encoder module."""
        self._encoder = module
        return self

    def with_compressor(self, module: nn.Module) -> "ModelBuilder":
        """Custom compressor module."""
        self._compressor = module
        return self

    def with_bank(self, module: nn.Module) -> "ModelBuilder":
        """Custom bank module."""
        self._bank = module
        return self

    def with_depth_stages(self, stages) -> "ModelBuilder":
        """Custom depth layers (module, list, or ReasoningDepth)."""
        self._depth = stages
        return self

    def build(self) -> WideNDepth:
        """Construct the WideNDepth model."""
        return _construct_model(
            self.config,
            wide=self._wide,
            encoder=self._encoder,
            compressor=self._compressor,
            bank=self._bank,
            depth=self._depth,
        )


@dataclass
class TrainingMetrics:
    """Structured return value for train_epoch / evaluate."""
    avg_loss: float
    total_loss: float
    samples: int
    steps: int


class Trainer:
    """Foundation training loop for tensor-to-tensor models."""

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        device: torch.device | str | None = None,
        amp: bool = True,
        grad_accumulation: int = 1,
        clip_grad: float | None = None,
        amp_dtype: torch.dtype | None = None,
        profile: HardwareProfile | None = None,
        backend: OptimizedBackend | None = None,
    ):
        if grad_accumulation < 1:
            raise ValueError("grad_accumulation must be positive")
        if clip_grad is not None and clip_grad <= 0:
            raise ValueError("clip_grad must be positive")
        if profile is not None and backend is not None:
            raise TypeError("pass either profile or backend, not both")
        if loss is not None and loss_fn is not None:
            raise TypeError("pass either loss or loss_fn, not both")

        self.loss_fn = loss_fn if loss_fn is not None else loss
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Create backend with user-specified device/dtype if provided
        if backend is not None:
            self.backend = backend
        elif profile is not None:
            self.backend = OptimizedBackend(profile)
        elif device is not None and str(device).startswith("cpu"):
            from winc.runtime import HardwareProfile
            self.backend = OptimizedBackend(
                HardwareProfile(torch.device("cpu"), torch.float32)
            )
        else:
            self.backend = OptimizedBackend(HardwareProfile(self.device, torch.float32))
        self.profile = self.backend.profile
        self.model = self.backend.prepare(model)
        self.optimizer = optimizer or self.backend.optimizer(self._default_parameters(model))
        self.grad_accumulation = grad_accumulation
        self.clip_grad = clip_grad
        self.amp = amp and self.backend.scaler_enabled
        self.amp_dtype = amp_dtype if amp_dtype is not None else torch.bfloat16
        self.scaler = (
            torch.amp.GradScaler()
            if self.amp and self.amp_dtype == torch.float16
            else None
        )

    def _default_parameters(self, model: nn.Module):
        return model.parameters()

    def _context(self):
        if self.amp:
            return torch.autocast(self.device.type, dtype=self.amp_dtype)
        from contextlib import nullcontext
        return nullcontext()

    def _transfer(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device)

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        """Single training step (no accumulation). Returns loss value."""
        self.model.train()
        inputs, targets = batch
        inputs = self._transfer(inputs)
        targets = self._transfer(targets)
        with self._context():
            prediction = self.model(inputs)
            loss = self.loss_fn(prediction, targets) / self.grad_accumulation
        loss.backward()
        if self.clip_grad is not None:
            self.backend.clip_grad_norm(self.model, self.clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(loss.detach()) * self.grad_accumulation

    def train_epoch(self, batches: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> TrainingMetrics:
        """Run one full epoch. Returns aggregate metrics."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total, steps = 0.0, 0
        count = 0
        for count, batch in enumerate(batches, 1):
            inputs, targets = batch
            inputs = self._transfer(inputs)
            targets = self._transfer(targets)
            with self._context():
                prediction = self.model(inputs)
                loss = self.loss_fn(prediction, targets) / self.grad_accumulation
            loss.backward()
            if self.clip_grad is not None:
                self.backend.clip_grad_norm(self.model, self.clip_grad)
            if count % self.grad_accumulation == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                steps += 1
            total += float(loss.detach()) * self.grad_accumulation
        if count and count % self.grad_accumulation:
            if self.clip_grad is not None:
                self.backend.clip_grad_norm(self.model, self.clip_grad)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            steps += 1
        return TrainingMetrics(
            avg_loss=total / max(count, 1),
            total_loss=total,
            samples=count,
            steps=steps,
        )

    def evaluate(self, batches: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> TrainingMetrics:
        """Evaluate on batches without gradient updates."""
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for inputs, targets in batches:
                inputs = self._transfer(inputs)
                targets = self._transfer(targets)
                with self._context():
                    prediction = self.model(inputs)
                    loss = self.loss_fn(prediction, targets)
                total += float(loss)
                count += 1
        return TrainingMetrics(
            avg_loss=total / max(count, 1),
            total_loss=total,
            samples=count,
            steps=0,
        )


@dataclass
class Regularizer:
    """Composable loss component.

    ``compute`` receives an aux dict (e.g. {"branches": ..., "bank": ...})
    and returns a scalar tensor to be added to the loss.
    """
    weight: float
    compute: Callable[[dict[str, Any]], torch.Tensor]
    name: str


class Orthogonality(Regularizer):
    """Encourage orthogonal branches."""

    def __init__(self, weight: float = 0.1):
        super().__init__(
            weight=weight,
            compute=lambda aux: orthogonality_loss(aux["branches"]),
            name="orthogonality",
        )


class DiverseBank(Regularizer):
    """Encourage diverse bank activations."""

    def __init__(self, weight: float = 0.05):
        super().__init__(
            weight=weight,
            compute=lambda aux: diversity_loss(aux["bank"]),
            name="diverse_bank",
        )


class RegularizedTrainer(Trainer):
    """Trainer with optional regularization losses."""

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        regularizers: list[Regularizer] | None = None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        **base_kwargs,
    ):
        super().__init__(
            model, loss=loss, optimizer=optimizer, loss_fn=loss_fn, **base_kwargs
        )
        self.regularizers = regularizers or []
        for reg in self.regularizers:
            if isinstance(reg, DiverseBank):
                if hasattr(self.model, "bank") and hasattr(self.model.bank, "detach"):
                    self.model.bank.detach = False

    def _compute_regularization(self, aux: dict) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.device)
        for reg in self.regularizers:
            try:
                total = total + reg.weight * reg.compute(aux)
            except (KeyError, TypeError) as exc:
                import warnings
                warnings.warn(
                    f"Regularizer '{reg.name}' raised {type(exc).__name__}: {exc}. "
                    "Check aux keys and regularizer inputs.",
                    stacklevel=2,
                )
        return total

    def _forward_with_aux(self, inputs: torch.Tensor):
        if hasattr(self.model, "forward") and "return_aux" in inspect.signature(self.model.forward).parameters:
            return self.model(inputs, return_aux=True)
        return self.model(inputs), {}

    def train_epoch(self, batches: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> TrainingMetrics:
        """Train one epoch with optional regularization losses."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total, count, steps = 0.0, 0, 0
        for count, batch in enumerate(batches, 1):
            inputs, targets = batch
            inputs = self._transfer(inputs)
            targets = self._transfer(targets)
            with self._context():
                prediction, aux = self._forward_with_aux(inputs)
                loss = self.loss_fn(prediction, targets)
                reg_loss = self._compute_regularization(aux)
                loss = (loss + reg_loss) / self.grad_accumulation
            loss.backward()
            if self.clip_grad is not None:
                self.backend.clip_grad_norm(self.model, self.clip_grad)
            if count % self.grad_accumulation == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                steps += 1
            total += float(loss.detach()) * self.grad_accumulation
        if count and count % self.grad_accumulation:
            if self.clip_grad is not None:
                self.backend.clip_grad_norm(self.model, self.clip_grad)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            steps += 1
        return TrainingMetrics(
            avg_loss=total / max(count, 1),
            total_loss=total,
            samples=count,
            steps=steps,
        )


class TrainingContext:
    """Orchestrates checkpoint/logging/device around a Trainer."""

    def __init__(
        self,
        trainer: Trainer,
        *,
        checkpoint_dir: Path | None = None,
        log_every: int = 100,
        device: torch.device | str | None = None,
    ):
        self.trainer = trainer
        self.checkpoint_dir = checkpoint_dir
        self.log_every = log_every
        self._on_step_end = None
        self._on_epoch_end = None
        self._on_optimizer_step = None
        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def on_step_end(self, callback: Callable):
        self._on_step_end = callback
        return callback

    def on_epoch_end(self, callback: Callable):
        self._on_epoch_end = callback
        return callback

    def on_optimizer_step(self, callback: Callable):
        self._on_optimizer_step = callback
        return callback

    def train_epoch(self, batches: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> TrainingMetrics:
        metrics = self.trainer.train_epoch(batches)
        if self._on_epoch_end is not None:
            self._on_epoch_end(metrics)
        return metrics

    def save_checkpoint(self, path: str | Path) -> None:
        from ..language.checkpoint import save_checkpoint
        save_checkpoint(self.trainer.model, self.trainer.optimizer, Path(path))

    def load_checkpoint(self, path: str | Path) -> None:
        from ..language.checkpoint import load_checkpoint
        load_checkpoint(Path(path), self.trainer.model, self.trainer.optimizer)

    def __repr__(self):
        return f"TrainingContext(checkpoint_dir={self.checkpoint_dir!r})"


# Backward-compatible KnowledgeTrainer: thin wrapper over RegularizedTrainer
class KnowledgeTrainer(RegularizedTrainer):
    """Backward-compatible training wrapper for knowledge-path regularization.

    Equivalent to::

        RegularizedTrainer(
            model, loss, optimizer,
            regularizers=[
                Orthogonality(orthogonal),
                DiverseBank(diverse_bank),
            ],
        )
    """

    def __init__(
        self,
        model: WideNDepth,
        optimizer: torch.optim.Optimizer | None,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        orthogonal: float = 0.0,
        diverse_bank: float = 0.0,
        profile: HardwareProfile | None = None,
        backend: OptimizedBackend | None = None,
        amp: bool = True,
        grad_accumulation: int = 1,
        clip_grad: float | None = None,
        amp_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        if orthogonal < 0 or diverse_bank < 0:
            raise ValueError("regularization weights must be non-negative")

        regularizers = []
        if orthogonal > 0:
            regularizers.append(Orthogonality(orthogonal))
        if diverse_bank > 0:
            regularizers.append(DiverseBank(diverse_bank))

        super().__init__(
            model,
            loss=loss,
            optimizer=optimizer,
            regularizers=regularizers,
            profile=profile,
            backend=backend,
            amp=amp,
            grad_accumulation=grad_accumulation,
            clip_grad=clip_grad,
            amp_dtype=amp_dtype,
            device=device,
        )

    def _default_parameters(self, model: WideNDepth):
        return model.knowledge_parameters()

    def step(self, batches: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Backward-compatible step() — runs a **full epoch**, not a single step.

        Despite the name, this consumes the entire ``batches`` iterable and
        returns the mean loss. Do not call this inside your own epoch loop or
        training will run N² effective passes. Prefer ``train_epoch()`` directly.
        """
        metrics = self.train_epoch(batches)
        return metrics.avg_loss

def configure(config: Config, **changes) -> Config:
    """Deprecated: use config.evolve(**changes) instead."""
    warnings.warn(
        "configure() is deprecated; use config.evolve(**changes) instead.",
        UserWarning,
        stacklevel=2,
    )
    return config.evolve(**changes)
