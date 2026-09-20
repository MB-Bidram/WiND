"""Architecture-aware LoRA adapters for WiND models.

The adapter layer lives in WiND because target semantics (wide, bank, depth,
decoder) belong to the model architecture.  It uses ordinary PyTorch modules
and parametrizations, so WiNC remains responsible only for execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize


@dataclass(frozen=True)
class LoRAConfig:
    """Default adapter policy and optional region/name overrides.

    ``regions`` uses semantic WiND names: ``wide``, ``pkm``, ``encoder``,
    ``compressor``, ``bank``, ``depth``, ``alpha``, ``decoder``, ``lm_head``,
    ``embeddings``, and ``custom``.  ``include``/``exclude`` are stable
    fnmatch patterns over fully-qualified module or parameter names.
    """

    rank: int = 8
    alpha: float | None = None
    dropout: float = 0.0
    initialization: str = "lora"  # lora (zero delta) | gaussian | zeros
    adapter: str = "lora"  # lora | dora | pkm_value
    enabled: bool = True
    bias: str = "none"  # none | all | adapted
    regions: tuple[str, ...] = ()
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    overrides: tuple[dict[str, Any], ...] = ()
    adapt_parameters: bool = True
    freeze_base: bool = True

    def __post_init__(self):
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("rank must be a positive integer")
        if self.alpha is not None and self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.bias not in {"none", "all", "adapted"}:
            raise ValueError("bias must be 'none', 'all', or 'adapted'")
        if self.initialization not in {"lora", "gaussian", "zeros"}:
            raise ValueError("initialization must be 'lora', 'gaussian', or 'zeros'")
        if self.adapter not in {"lora", "dora", "pkm_value"}:
            raise ValueError("adapter must be 'lora', 'dora', or 'pkm_value'")

    @property
    def scale(self) -> float:
        return (self.alpha if self.alpha is not None else self.rank) / self.rank


def _init_pair(a: torch.Tensor, b: torch.Tensor, mode: str) -> None:
    if mode == "zeros":
        nn.init.zeros_(a); nn.init.zeros_(b)
    elif mode == "gaussian":
        nn.init.normal_(a, std=0.02); nn.init.normal_(b, std=0.02)
    else:
        nn.init.kaiming_uniform_(a, a=5 ** 0.5); nn.init.zeros_(b)


@dataclass(frozen=True)
class AdapterTarget:
    name: str
    region: str
    kind: str
    module_type: str
    shape: tuple[int, ...]
    supported: bool
    reason: str = ""


def _region(name: str, module: nn.Module | None = None) -> str:
    first = name.split(".", 1)[0]
    if name == "alpha" or first == "alpha": return "alpha"
    if first == "wide":
        return "pkm" if ".pkm" in name or (module is not None and "PKM" in type(module).__name__) else "wide"
    if first in {"encoder", "compressor", "bank", "depth", "decoder", "lm_head"}: return first
    if first in {"embedding", "source_position", "target_position"}: return "embeddings"
    return "custom"


def _match(name: str, region: str, config: LoRAConfig) -> bool:
    allowed = not config.regions or region in config.regions
    included = not config.include or any(fnmatchcase(name, pattern) for pattern in config.include)
    excluded = any(fnmatchcase(name, pattern) for pattern in config.exclude)
    return allowed and included and not excluded


class LoRALinear(nn.Module):
    """Contract-preserving additive low-rank wrapper for ``nn.Linear``."""
    def __init__(self, base: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.base = base
        self.rank, self.scaling = config.rank, config.scale
        self.enabled = config.enabled
        self.dropout = nn.Dropout(config.dropout) if config.dropout else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, config.rank, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B = nn.Linear(config.rank, base.out_features, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        _init_pair(self.lora_A.weight, self.lora_B.weight, config.initialization)
        self.merged = False

    def delta_weight(self) -> torch.Tensor:
        return self.lora_B.weight @ self.lora_A.weight * self.scaling

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight.add_(self.delta_weight().to(self.base.weight.dtype)); self.merged = True
        return self

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight.sub_(self.delta_weight().to(self.base.weight.dtype)); self.merged = False
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        return output if self.merged or not self.enabled else output + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

class DoRALinear(LoRALinear):
    """Weight-decomposed low-rank adaptation for linear projections."""
    def __init__(self, base: nn.Linear, config: LoRAConfig):
        if config.dropout:
            raise ValueError("DoRA requires dropout=0 to preserve mergeable weight semantics")
        super().__init__(base, config)
        self.lora_magnitude = nn.Parameter(base.weight.detach().float().norm(dim=1).to(base.weight.dtype))
        self._merge_reference: torch.Tensor | None = None

    def adapted_weight(self) -> torch.Tensor:
        direction = self.base.weight + self.delta_weight().to(self.base.weight.dtype)
        norm = direction.float().norm(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
        return direction * (self.lora_magnitude.float().unsqueeze(1) / norm).to(direction.dtype)

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self._merge_reference = self.base.weight.detach().clone()
            self.base.weight.copy_(self.adapted_weight().to(self.base.weight.dtype))
            self.merged = True
        return self

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            if self._merge_reference is None:
                raise RuntimeError("cannot unmerge DoRA without its original base weight")
            self.base.weight.copy_(self._merge_reference)
            self._merge_reference = None
            self.merged = False
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) if self.merged or not self.enabled else F.linear(x, self.adapted_weight(), self.base.bias)

class LoRAEmbedding(nn.Module):
    """Embedding LoRA: ``embedding(ids, A) @ B`` added to base lookup."""
    def __init__(self, base: nn.Embedding, config: LoRAConfig):
        super().__init__()
        self.base = base; self.rank, self.scaling = config.rank, config.scale
        self.enabled = config.enabled
        self.dropout = nn.Dropout(config.dropout) if config.dropout else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(base.num_embeddings, config.rank, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(config.rank, base.embedding_dim, device=base.weight.device, dtype=base.weight.dtype))
        _init_pair(self.lora_A, self.lora_B, config.initialization)
        self.merged = False

    def delta_weight(self): return self.lora_A @ self.lora_B * self.scaling
    @torch.no_grad()
    def merge(self):
        if not self.merged: self.base.weight.add_(self.delta_weight()); self.merged = True
        return self
    @torch.no_grad()
    def unmerge(self):
        if self.merged: self.base.weight.sub_(self.delta_weight()); self.merged = False
        return self
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        output = self.base(ids)
        if self.merged or not self.enabled: return output
        adapter = F.embedding(ids, self.lora_A, self.base.padding_idx, None, 2.0, False, False)
        return output + self.dropout(adapter) @ self.lora_B * self.scaling


class LoRAParameterization(nn.Module):
    """Low-rank parameter delta for matrices, PKM tables, and scalar Alpha."""
    def __init__(self, parameter: torch.Tensor, config: LoRAConfig, kind: str):
        super().__init__()
        self.kind, self.scaling, self.merged = kind, config.scale, False
        self.enabled = config.enabled
        if parameter.ndim == 0:
            self.lora_delta = nn.Parameter(torch.zeros_like(parameter))
            if config.initialization == "gaussian": nn.init.normal_(self.lora_delta, std=0.02)
            return
        if parameter.ndim == 2:
            rows, cols = parameter.shape
            self.lora_A = nn.Parameter(torch.empty(rows, config.rank, device=parameter.device, dtype=parameter.dtype))
            self.lora_B = nn.Parameter(torch.zeros(config.rank, cols, device=parameter.device, dtype=parameter.dtype))
            _init_pair(self.lora_A, self.lora_B, config.initialization); return
        if kind == "pkm_keys" and parameter.ndim == 4:
            h, f, rows, cols = parameter.shape
            self.lora_A = nn.Parameter(torch.empty(h, f, rows, config.rank, device=parameter.device, dtype=parameter.dtype))
            self.lora_B = nn.Parameter(torch.zeros(h, f, config.rank, cols, device=parameter.device, dtype=parameter.dtype))
            _init_pair(self.lora_A, self.lora_B, config.initialization); return
        if kind == "pkm_values" and parameter.ndim == 4:
            h, rows, f, cols = parameter.shape
            self.lora_A = nn.Parameter(torch.empty(h, f, rows, config.rank, device=parameter.device, dtype=parameter.dtype))
            self.lora_B = nn.Parameter(torch.zeros(h, f, config.rank, cols, device=parameter.device, dtype=parameter.dtype))
            _init_pair(self.lora_A, self.lora_B, config.initialization); return
        raise ValueError(f"ordinary LoRA does not support parameter kind={kind!r}, shape={tuple(parameter.shape)}")

    def delta(self) -> torch.Tensor:
        if hasattr(self, "lora_delta"): return self.lora_delta
        value = self.lora_A @ self.lora_B * self.scaling
        return value.permute(0, 2, 1, 3) if self.kind == "pkm_values" else value
    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return original if self.merged or not self.enabled else original + self.delta().to(original.dtype)
    @torch.no_grad()
    def merge(self, original: torch.Tensor):
        if not self.merged: original.add_(self.delta().to(original.dtype)); self.merged = True
    @torch.no_grad()
    def unmerge(self, original: torch.Tensor):
        if self.merged: original.sub_(self.delta().to(original.dtype)); self.merged = False


class DoRAParameterization(LoRAParameterization):
    """DoRA for two-dimensional non-module parameters, e.g. learned queries."""
    def __init__(self, parameter: torch.Tensor, config: LoRAConfig, kind: str):
        if parameter.ndim != 2:
            raise ValueError("DoRA parameterization supports matrices only")
        if config.dropout:
            raise ValueError("DoRA requires dropout=0 to preserve mergeable weight semantics")
        super().__init__(parameter, config, kind)
        self.lora_magnitude = nn.Parameter(parameter.detach().float().norm(dim=1).to(parameter.dtype))
        self._merge_reference: torch.Tensor | None = None

    def adapted(self, original: torch.Tensor) -> torch.Tensor:
        direction = original + self.delta().to(original.dtype)
        norm = direction.float().norm(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
        return direction * (self.lora_magnitude.float().unsqueeze(1) / norm).to(direction.dtype)

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return original if self.merged or not self.enabled else self.adapted(original)

    @torch.no_grad()
    def merge(self, original: torch.Tensor):
        if not self.merged:
            self._merge_reference = original.detach().clone()
            original.copy_(self.adapted(original).to(original.dtype)); self.merged = True

    @torch.no_grad()
    def unmerge(self, original: torch.Tensor):
        if self.merged:
            if self._merge_reference is None:
                raise RuntimeError("cannot unmerge DoRA without its original parameter")
            original.copy_(self._merge_reference); self._merge_reference = None; self.merged = False


class PKMValueLoRA(LoRAParameterization):
    """Selected-slot LoRA for FlashPKM values ``[H, M, F, V]``.

    PKM selection depends only on keys. This adapter therefore computes value
    deltas only for selected memory slots instead of materializing a complete
    table update on every forward.
    """
    def __init__(self, parameter: torch.Tensor, config: LoRAConfig, kind: str = "pkm_values"):
        if kind != "pkm_values" or parameter.ndim != 4:
            raise ValueError("PKMValueLoRA requires a [heads, memory, factors, value] table")
        super().__init__(parameter, config, kind)

    def gather_values(self, head_indices: torch.Tensor,
                      selected_memory_flat: torch.Tensor) -> torch.Tensor:
        """Return summed selected-slot deltas as ``[rows, topk, value]``."""
        if self.merged or not self.enabled:
            return self.lora_B.new_zeros((*selected_memory_flat.shape[:2], self.lora_B.size(-1)))
        pieces = []
        for factor in range(selected_memory_flat.size(-1)):
            a = self.lora_A[head_indices[:, None], factor, selected_memory_flat[:, :, factor]]
            b = self.lora_B[head_indices, factor]
            pieces.append(torch.matmul(a, b))
        return torch.stack(pieces, dim=0).sum(dim=0) * self.scaling


def discover_lora_targets(model: nn.Module) -> list[AdapterTarget]:
    """List adapter-compatible operations and semantic regions without mutation."""
    results: list[AdapterTarget] = []
    module_parameter_ids: set[int] = set()
    for name, module in model.named_modules(remove_duplicate=False):
        if not name or isinstance(module, (LoRALinear, LoRAEmbedding)): continue
        if isinstance(module, nn.Linear):
            results.append(AdapterTarget(name, _region(name, module), "linear", type(module).__name__, tuple(module.weight.shape), True))
            module_parameter_ids.update(id(p) for p in module.parameters(recurse=False)); continue
        if isinstance(module, nn.Embedding):
            results.append(AdapterTarget(name, _region(name, module), "embedding", type(module).__name__, tuple(module.weight.shape), True))
            module_parameter_ids.update(id(p) for p in module.parameters(recurse=False))
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if (id(parameter) in module_parameter_ids or ".parametrizations." in name
                or "lora_" in name or "_wind_lora_parameter_adapters" in name):
            continue
        parent_name, _, attr = name.rpartition("."); owner = model.get_submodule(parent_name) if parent_name else model
        kind = "alpha" if name == "alpha" else "pkm_keys" if name.endswith(".keys") else "pkm_values" if name.endswith(".values") else "matrix" if parameter.ndim == 2 else "unsupported"
        supported = kind != "unsupported"
        reason = "" if supported else "ordinary additive LoRA supports scalar, matrix, PKM key, and PKM value parameters only"
        results.append(AdapterTarget(name, _region(name, owner), kind, type(owner).__name__, tuple(parameter.shape), supported, reason))
    return results


def _parent_and_key(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parent_path, _, key = path.rpartition(".")
    return (root.get_submodule(parent_path) if parent_path else root), key


def _replace(root: nn.Module, path: str, value: nn.Module) -> None:
    parent, key = _parent_and_key(root, path)
    parent._modules[key] = value


def _rule_config(base: LoRAConfig, name: str, region: str) -> LoRAConfig | None:
    selected = base if _match(name, region, base) else None
    for override in base.overrides:
        trial = LoRAConfig(**{**asdict(base), **override, "overrides": ()})
        if _match(name, region, trial): selected = trial
    return selected


def inject_lora(model: nn.Module, config: LoRAConfig | None = None) -> list[AdapterTarget]:
    """Inject adapters selected by regions/names, returning applied targets."""
    config = config or LoRAConfig()
    targets = discover_lora_targets(model); applied: list[AdapterTarget] = []
    # Module aliases are replaced with one wrapper object, preserving shared modules.
    aliases: dict[int, list[str]] = {}
    modules: dict[int, nn.Module] = {}
    for name, module in model.named_modules(remove_duplicate=False):
        if name: aliases.setdefault(id(module), []).append(name); modules[id(module)] = module
    for target in targets:
        if target.kind not in {"linear", "embedding"}: continue
        cfg = _rule_config(config, target.name, target.region)
        if cfg is None: continue
        module = model.get_submodule(target.name)
        if isinstance(module, (LoRALinear, LoRAEmbedding)): continue
        if target.kind == "linear":
            if cfg.adapter == "pkm_value":
                continue
            wrapper = DoRALinear(module, cfg) if cfg.adapter == "dora" else LoRALinear(module, cfg)
        else:
            if cfg.adapter != "lora":
                continue
            wrapper = LoRAEmbedding(module, cfg)
        for alias in aliases[id(module)]: _replace(model, alias, wrapper)
        applied.append(target)
    if config.adapt_parameters:
        for target in discover_lora_targets(model):
            if target.kind in {"linear", "embedding"} or not target.supported: continue
            cfg = _rule_config(config, target.name, target.region)
            if cfg is None: continue
            parent_path, _, attr = target.name.rpartition("."); owner = model.get_submodule(parent_path) if parent_path else model
            if parametrize.is_parametrized(owner, attr): continue
            parameter = getattr(owner, attr)
            if cfg.adapter == "lora":
                parametrization = LoRAParameterization(parameter, cfg, target.kind)
            elif cfg.adapter == "dora" and target.kind == "matrix":
                parametrization = DoRAParameterization(parameter, cfg, target.kind)
            elif cfg.adapter == "pkm_value" and target.kind == "pkm_values":
                parametrization = PKMValueLoRA(parameter, cfg, target.kind)
            else:
                continue
            # ``keys`` and ``values`` collide with ModuleDict's public methods
            # inside torch parametrizations. FactorizedPKM exposes a tiny
            # execution hook for those tables, preserving original parameter
            # names and state-dict compatibility.
            if target.kind in {"pkm_keys", "pkm_values"}:
                adapters = getattr(owner, "_wind_lora_parameter_adapters", None)
                if adapters is None:
                    adapters = nn.ModuleDict(); owner.add_module("_wind_lora_parameter_adapters", adapters)
                adapters[f"p_{attr}"] = parametrization
            else:
                parametrize.register_parametrization(owner, attr, parametrization)
            applied.append(target)
    if config.freeze_base: mark_only_lora_trainable(model, bias=config.bias)
    return applied


def iter_lora_modules(model: nn.Module) -> Iterable[nn.Module]:
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAEmbedding, LoRAParameterization,
                               DoRALinear, DoRAParameterization, PKMValueLoRA)): yield module


def mark_only_lora_trainable(model: nn.Module, *, bias: str = "none") -> None:
    """Freeze base parameters and select only adapter (and requested bias) params."""
    if bias not in {"none", "all", "adapted"}: raise ValueError("invalid bias policy")
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(True)
    if bias == "all":
        for name, parameter in model.named_parameters():
            if name.endswith("bias"): parameter.requires_grad_(True)
    elif bias == "adapted":
        for module in model.modules():
            if isinstance(module, LoRALinear) and module.base.bias is not None: module.base.bias.requires_grad_(True)


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in model.parameters() if p.requires_grad)


def lora_parameter_report(model: nn.Module) -> dict[str, Any]:
    regions: dict[str, int] = {}
    names: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            names.append(name)
            if name.startswith("parametrizations.alpha."):
                region = "alpha"
            else:
                region = _region(name.replace(".parametrizations.", "."))
            regions[region] = regions.get(region, 0) + parameter.numel()
    return {"total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "regions": regions, "trainable_names": names}


@torch.no_grad()
def merge_lora(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAEmbedding, DoRALinear)): module.merge()
    for module in model.modules():
        if isinstance(module, nn.Module) and hasattr(module, "parametrizations"):
            for attr, items in getattr(module, "parametrizations", {}).items():
                for item in items:
                    if isinstance(item, (LoRAParameterization, DoRAParameterization)):
                        item.merge(module.parametrizations[attr].original)
        adapters = getattr(module, "_wind_lora_parameter_adapters", None)
        if adapters is not None:
            for key, item in adapters.items():
                if isinstance(item, (LoRAParameterization, PKMValueLoRA)):
                    item.merge(getattr(module, key[2:]))


@torch.no_grad()
def unmerge_lora(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAEmbedding, DoRALinear)): module.unmerge()
    for module in model.modules():
        if isinstance(module, nn.Module) and hasattr(module, "parametrizations"):
            for attr, items in getattr(module, "parametrizations", {}).items():
                for item in items:
                    if isinstance(item, (LoRAParameterization, DoRAParameterization)):
                        item.unmerge(module.parametrizations[attr].original)
        adapters = getattr(module, "_wind_lora_parameter_adapters", None)
        if adapters is not None:
            for key, item in adapters.items():
                if isinstance(item, (LoRAParameterization, PKMValueLoRA)):
                    item.unmerge(getattr(module, key[2:]))


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()
            if "lora_" in name or name.endswith(".merged")}


def save_lora_adapters(model: nn.Module, path: str | Path, config: LoRAConfig | None = None) -> None:
    torch.save({"format": "wind.lora", "version": 1, "config": asdict(config) if config else None,
                "state": adapter_state_dict(model)}, Path(path))


def load_lora_adapters(model: nn.Module, path: str | Path, *, strict: bool = True) -> None:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != "wind.lora": raise ValueError("not a WiND LoRA adapter checkpoint")
    missing, unexpected = model.load_state_dict(payload["state"], strict=False)
    relevant_missing = [key for key in missing if "lora_" in key]
    if strict and (relevant_missing or unexpected):
        raise ValueError(f"adapter checkpoint mismatch: missing={relevant_missing}, unexpected={unexpected}")


def remove_lora(model: nn.Module, *, merge: bool = False) -> None:
    """Remove wrappers/parametrizations, optionally first merging deltas."""
    if merge: merge_lora(model)
    for name, module in list(model.named_modules()):
        if not name: continue
        if isinstance(module, (LoRALinear, LoRAEmbedding, DoRALinear)): _replace(model, name, module.base)
    for module in model.modules():
        if hasattr(module, "parametrizations"):
            for attr in list(module.parametrizations.keys()):
                if any(isinstance(x, (LoRAParameterization, DoRAParameterization)) for x in module.parametrizations[attr]):
                    parametrize.remove_parametrizations(module, attr, leave_parametrized=merge)
        adapters = getattr(module, "_wind_lora_parameter_adapters", None)
        if adapters is not None and all(isinstance(item, (LoRAParameterization, PKMValueLoRA)) for item in adapters.values()):
            del module._modules["_wind_lora_parameter_adapters"]
