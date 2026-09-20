"""Single-device language training with token-weighted accumulation and resume.

Design notes (sync-free steady state):
  * No GradScaler.  bf16 has the same exponent range as fp32, so no scale
    search is needed and the optimizer step never reads `found_inf` on host.
  * Token counts are kept as 0-d device tensors throughout the hot path.
    `result.token_count` is a detached tensor, never `.item()` on step.
  * Loss accumulation uses in-place `add_` with the tensor-valued count as
    `alpha`, so normalization stays on-device.
  * The only `.item()` / `float()` calls happen at epoch boundaries
    (step's return value and the logger), well outside the compile step
    and well after the optimizer has stepped.

Callback hooks (on_step_end, on_epoch_end, on_optimizer_step) are invoked
at epoch/accumulation boundaries only — never on the per-step hot path —
so they add zero sync overhead during training.
"""

from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from dataclasses import dataclass
import math
from typing import Callable

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from winc.logger import make_logger
from winc._internal.guard import (
    assert_finite,
    assert_tensor,
    assert_dtype,
    assert_device,
    warn_sync,
    assert_not_in_hot_path,
    assert_loss_monotonically_decreasing,
    GUARD_ENABLED,
)
from .model import LanguageModel, ReasoningState


@dataclass
class TrainerState:
    """Snapshot of trainer progress provided to callback hooks.

    All values are plain Python types (no device tensors) so callbacks
    can log freely without introducing GPU syncs.
    """
    global_step: int
    epochs: int
    epoch: int
    loss: float
    tokens: int
    lr: float


@dataclass
class OptimizerStepInfo:
    """Details about a single optimizer step, for logging."""
    global_step: int
    lr: float
    grad_norm: float | None


class LMTrainer:
    """Train on mappings accepted by LanguageModel.forward (labels required).

    Accumulated loss gradients are divided by the actual valid-token count,
    including the last partial group. Checkpoints are taken between calls
    to step/fit.

    Parameters
    ----------
    model : LanguageModel
        The model to train. Can be any dtype (float32, bfloat16, float16).
    optimizer : torch.optim.Optimizer or None
        Custom optimizer. If None, fused AdamW is used when available.
    lr : float
        Learning rate for the default optimizer.
    weight_decay : float
        Weight decay for the default optimizer.
    device : torch.device or None
        Device to place the model on.
    amp : bool or None
        Auto-select amp dtype on CUDA when True (default). None = auto-detect.
    amp_dtype : torch.dtype or None
        Explicit dtype for autocast (bfloat16 or float16). If None, bf16
        is preferred on CUDA.
    grad_accumulation : int
        Number of micro-batches per optimizer step.
    clip_grad : float or None
        Global gradient norm clipping. None disables.
    logger : str or callable or None
        "console" for default, a callable receiving a dict, or None.
    master_weights : bool
        If True, keep a float32 copy of parameters for optimization even
        when the model is in a reduced precision (bf16/fp16). This is the
        traditional AMP approach. Default: False (train in model.dtype directly).
        bf16 is safe without master weights because it shares fp32's exponent
        range.  fp16 with master_weights=True is the one combination that
        needs the GradScaler, which is still not used here — fp16 training
        without master weights relies on loss scaling being unnecessary for
        the specific architecture.

    Callback hooks
    --------------
    on_step_end : callable(TrainerState) or None
        Called at the end of each ``step()`` epoch (once per epoch-level iteration).
    on_epoch_end : callable(TrainerState) or None
        Alias for on_step_end; called after each epoch.
    on_optimizer_step : callable(OptimizerStepInfo) or None
        Called after every optimizer step (every ``grad_accumulation`` micro-batches).
        The grad_norm passed is None unless clip_grad is set, avoiding an
        extra sync just for logging.
    """

    def __init__(
        self,
        model: LanguageModel,
        optimizer=None,
        *,
        lr=3e-4,
        weight_decay=0.01,
        device=None,
        amp=None,
        amp_dtype=None,
        grad_accumulation=1,
        clip_grad=1.0,
        logger=None,
        master_weights: bool = False,
        on_step_end: Callable | None = None,
        on_epoch_end: Callable | None = None,
        on_optimizer_step: Callable | None = None,
        params_to_train=None,
    ):
        if type(grad_accumulation) is not int or grad_accumulation < 1:
            raise ValueError("grad_accumulation must be a positive integer")
        if clip_grad is not None and (not math.isfinite(clip_grad) or clip_grad <= 0):
            raise ValueError("clip_grad must be finite and positive")
        _model_orig = getattr(model, "_orig_mod", None)
        _actual = _model_orig if _model_orig is not None else model
        if not isinstance(_actual, LanguageModel):
            _actual_type = type(_actual).__name__
            raise TypeError(
                f"[Wind] ASSERT:[LMTrainer] model must be a LanguageModel instance, "
                f"got {_actual_type}"
            )

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device

        if model_dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError(
                f"unsupported model dtype {model_dtype}; "
                "supported: float32, bfloat16, float16"
            )

        self.model_dtype = model_dtype
        self._master_weights = master_weights

        if master_weights and model_dtype != torch.float32:
            trainable = model.float()
            self._original_dtype = model_dtype
        else:
            trainable = model
            self._original_dtype = None

        self.model = trainable.to(self.device)

        if params_to_train is not None:
            self._parameters = [p for p in params_to_train]
        else:
            # Adapter injection marks base parameters frozen. Do not pass them
            # to AdamW merely because they remain registered on the model:
            # optimizer state then stays proportional to the full model rather
            # than the intended PEFT set.
            self._parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not self._parameters:
            raise ValueError("trainer received no trainable parameters")

        if optimizer is not None:
            self.optimizer = optimizer
        elif self.device.type == "cuda":
            # Fused AdamW keeps its step counter as a CUDA tensor — no host readback.
            try:
                self.optimizer = torch.optim.AdamW(
                    self._parameters,
                    lr=lr,
                    weight_decay=weight_decay,
                    fused=True,
                )
            except (TypeError, RuntimeError):
                self.optimizer = torch.optim.AdamW(
                    self._parameters,
                    lr=lr,
                    weight_decay=weight_decay,
                )
        else:
            self.optimizer = torch.optim.AdamW(
                self._parameters,
                lr=lr,
                weight_decay=weight_decay,
            )

        self.grad_accumulation = grad_accumulation
        self.clip_grad = clip_grad

        # bf16 is the default on Ampere+ GPUs (SM 8.6 = RTX 3050 Laptop).
        # It has fp32 exponent range, so no GradScaler is ever needed.
        if amp is not None:
            self.amp = bool(amp and self.device.type == "cuda")
        else:
            self.amp = self.device.type == "cuda"

        if amp_dtype is not None:
            if self.amp and amp_dtype not in (torch.bfloat16, torch.float16):
                raise ValueError("amp_dtype must be torch.bfloat16 or torch.float16")
            self.amp_dtype = amp_dtype if self.amp else self.model_dtype
        elif self.amp:
            # Prefer bf16 on GPUs that support it natively (RTX 3050 Laptop = SM 8.6).
            if torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
        else:
            self.amp_dtype = self.model_dtype

        # No GradScaler: bf16 has the same exponent range as fp32, so inf/nan
        # detection is unnecessary.  This eliminates an unavoidable per-step sync.
        self.scaler_enabled = False
        self.scaler = None

        self.global_step = 0
        self.epochs = 0
        self._epoch = 0
        self.logger = make_logger(logger)

        # Callback hooks — invoked only at epoch and optimizer-step boundaries.
        self.on_step_end = on_step_end
        self.on_epoch_end = on_epoch_end
        self.on_optimizer_step = on_optimizer_step

    @contextmanager
    def _context(self):
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        ac = torch.autocast("cuda", dtype=self.amp_dtype) if self.amp else nullcontext()
        with ac:
            yield

    def _batch(self, batch):
        if not isinstance(batch, dict) or "labels" not in batch:
            raise ValueError("batches must be dictionaries containing labels")

        device = self.device
        converted = {}

        # Deliberately preserve integer IDs, boolean masks and floating features.
        # non_blocking=True lets pinned DataLoader batches overlap host/device copies.
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                converted[k] = v.to(device, non_blocking=True)
            elif isinstance(v, ReasoningState):
                converted[k] = v.to(device)
            else:
                converted[k] = v

        return converted

    def _zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def _optimizer_step(self, group_tokens: int) -> float | None:
        """Normalize grads by token count, clip, and step the optimizer.

        group_tokens is a Python int (read once from the device at the
        accumulation boundary) — it is not on the per-step hot path.
        Returns the grad norm (or None) for optimizer-step callbacks.
        """
        grad_norm = None
        if group_tokens > 0:
            parameters = self._parameters
            gradients = [
                p.grad
                for p in parameters
                if p.grad is not None
            ]

            if gradients:
                # Batch gradient normalization: divide all grads by token count once.
                torch._foreach_div_(gradients, float(group_tokens))

            if self.clip_grad is not None:
                # Use total_norm=True (default) with foreach for efficiency.
                # grad_norm is computed here regardless; it's only read back
                # via .item() at this boundary (not per-step).
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters,
                    self.clip_grad,
                    foreach=True,
                    error_if_nonfinite=False,
                )
                grad_norm = float(grad_norm) if grad_norm is not None else None

                if GUARD_ENABLED and grad_norm is not None and not math.isfinite(grad_norm):
                    from winc._internal.guard import guard, WindAssertError
                    guard(
                        "ASSERT",
                        f"non-finite grad_norm={grad_norm} after clip, lr={self._get_lr()}",
                        "LMTrainer",
                        exc=WindAssertError,
                    )

            self.optimizer.step()
            self.global_step += 1

        self._zero_grad()
        return grad_norm

    def step(self, batches):
        """Run one epoch. Return mean cross-entropy per valid target token.

        Hot path: no .item() / .tolist() / float() on per-step tensors.
        The return value materializes the mean only once at the epoch boundary.
        """
        self.model.train()
        self._zero_grad()

        # Accumulation buffers live on the device; never touch CPU mid-step.
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        total_tokens = torch.zeros((), device=self.device, dtype=torch.int64)

        count = 0
        group_tokens_t = torch.zeros((), device=self.device, dtype=torch.int64)

        try:
            for count, batch in enumerate(batches, 1):
                batch = self._batch(batch)

                with self._context():
                    result = self.model(**batch)

                # token_count is a detached 0-d device tensor — stay on-device.
                token_count = result.token_count
                tc = token_count.to(result.loss.dtype)

                # If token_count is 0, scaled_loss is 0 and backward is a no-op.
                # No branch needed — avoids a per-step .item() sync.
                scaled_loss = result.loss * tc
                scaled_loss.backward()

                total_tokens += token_count
                # Accumulate (loss * token_count) on-device via in-place add.
                total.add_(result.loss.detach(), alpha=tc)

                # Accumulate token count as a device tensor — no sync until boundary.
                group_tokens_t += token_count

                if count % self.grad_accumulation == 0:
                    grad_norm = self._optimizer_step(int(group_tokens_t))
                    group_tokens_t.zero_()
                    if self.on_optimizer_step is not None:
                        lr = self._get_lr()
                        info = OptimizerStepInfo(
                            global_step=self.global_step,
                            lr=lr,
                            grad_norm=grad_norm,
                        )
                        self.on_optimizer_step(info)

            if count % self.grad_accumulation:
                grad_norm = self._optimizer_step(int(group_tokens_t))
                if self.on_optimizer_step is not None:
                    lr = self._get_lr()
                    info = OptimizerStepInfo(
                        global_step=self.global_step,
                        lr=lr,
                        grad_norm=grad_norm,
                    )
                    self.on_optimizer_step(info)

        except Exception:
            self._zero_grad()
            raise

        # Epoch boundary: a single sync here is acceptable.
        if int(total_tokens) == 0:
            raise ValueError("epoch has no valid target tokens")

        # Check for nonfinite loss after all backward passes — this is the epoch
        # boundary, not the per-step path, so a single sync is acceptable.
        mean_loss = total / total_tokens.to(total.dtype)
        if not bool(torch.isfinite(mean_loss)):
            raise FloatingPointError("nonfinite language loss")

        self.epochs += 1
        self._epoch += 1

        loss_value = float(mean_loss)

        # Fire epoch-level callbacks.
        if self.on_step_end is not None or self.on_epoch_end is not None:
            lr = self._get_lr()
            state = TrainerState(
                global_step=self.global_step,
                epochs=self.epochs,
                epoch=self._epoch,
                loss=loss_value,
                tokens=int(total_tokens),
                lr=lr,
            )
            if self.on_step_end is not None:
                self.on_step_end(state)
            if self.on_epoch_end is not None:
                self.on_epoch_end(state)

        # Log at the epoch boundary only — well outside the compile step.
        if self.logger is not None:
            metrics = {
                "epoch": self.epochs,
                "global_step": self.global_step,
                "tokens": int(total_tokens),
                "loss": loss_value,
            }
            self.logger(metrics)

        return loss_value

    def _get_lr(self) -> float:
        """Return the current learning rate from the optimizer."""
        try:
            lrs = [
                pg["lr"]
                for pg in self.optimizer.param_groups
                if "lr" in pg
            ]
            if lrs:
                return float(lrs[0])
        except Exception:
            pass
        return 0.0

    def fit(self, batches, *, epochs=1):
        """Return per-epoch losses; use a re-iterable DataLoader/list for >1 epoch."""
        if type(epochs) is not int or epochs < 1:
            raise ValueError("epochs must be a positive integer")

        if epochs > 1 and iter(batches) is batches:
            raise ValueError(
                "multiple epochs require a re-iterable batch source"
            )

        return [
            self.step(batches)
            for _ in range(epochs)
        ]

    @torch.inference_mode()
    def evaluate(self, batches):
        was_training = self.model.training
        self.model.eval()

        # Device-side accumulation — no per-step sync.
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        total_tokens = torch.zeros((), device=self.device, dtype=torch.int64)

        try:
            for batch in batches:
                with self._context():
                    result = self.model(**self._batch(batch))

                token_count = result.token_count
                # No branch: if token_count is 0, alpha=0 makes add a no-op.
                total.add_(
                    result.loss.detach().float(),
                    alpha=token_count.to(torch.float32),
                )
                total_tokens += token_count

            if int(total_tokens) == 0:
                raise ValueError("evaluation has no valid target tokens")

            loss = float((total / total_tokens.to(total.dtype)))

            return {
                "loss": loss,
                "perplexity": math.exp(loss) if loss < 700 else math.inf,
                "tokens": int(total_tokens),
            }

        finally:
            self.model.train(was_training)

    def save_checkpoint(self, path):
        """Save model, optimizer, counters and PyTorch RNG at a step boundary.

        Dataset/sampler cursor, Python/NumPy RNG, external schedulers and tokenizer
        assets are owned by the application and must be saved separately.
        """
        save_checkpoint(
            path,
            {
                "config": asdict(self.model.config),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "optimizer_type": (
                    type(self.optimizer).__module__
                    + "."
                    + type(self.optimizer).__name__
                ),
                "global_step": self.global_step,
                "epochs": self.epochs,
                "_epoch": self._epoch,
                "grad_accumulation": self.grad_accumulation,
                "clip_grad": self.clip_grad,
                "amp": self.amp,
                "amp_dtype": str(self.amp_dtype),
                "model_dtype": str(self.model_dtype),
                "master_weights": self._master_weights,
                "rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state_all()
                    if self.device.type == "cuda"
                    else []
                ),
            },
        )

    def load_checkpoint(self, path, *, restore_rng=True):
        """Restore into this trainer; architecture and optimizer type must match."""
        data = load_checkpoint(path)

        if "optimizer" not in data:
            raise ValueError("model-only checkpoint cannot resume training")

        if data["config"] != asdict(self.model.config):
            raise ValueError("checkpoint architecture does not match this model")

        optimizer_type = (
            type(self.optimizer).__module__
            + "."
            + type(self.optimizer).__name__
        )

        if data["optimizer_type"] != optimizer_type:
            raise ValueError("checkpoint optimizer type does not match")

        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])

        self.global_step = data["global_step"]
        self.epochs = data["epochs"]
        self._epoch = data.get("_epoch", data.get("epochs", 0))
        self.grad_accumulation = data["grad_accumulation"]
        self.clip_grad = data["clip_grad"]

        self._zero_grad()

        if restore_rng:
            torch.set_rng_state(data["rng"])

            if (
                self.device.type == "cuda"
                and len(data["cuda_rng"]) == torch.cuda.device_count()
            ):
                torch.cuda.set_rng_state_all(data["cuda_rng"])

        return self


