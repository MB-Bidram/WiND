"""WiND Language Model: prefix-LM with WideNDepth internals.

Only source tokens enter the knowledge/reasoning path. Target tokens enter a
causal decoder after a right shift, preventing teacher-forcing answer leakage.
The encoder is bidirectional; this is a conditional/prefix LM, not a decoder-only
LM trained by passing the same full sequence to both source and target.

Submodules:
    config  - LMConfig dataclass
    model   - LanguageModel, LMOutput, ReasoningState
    trainer - LMTrainer
    checkpoint - Versioned tensor checkpoints (shared infrastructure)
"""

from .checkpoint import load_checkpoint, save_checkpoint
from .config import LMConfig
from .model import LanguageModel, LMOutput, ReasoningState
from .trainer import LMTrainer

__all__ = [
    "LMConfig",
    "LanguageModel",
    "LMOutput",
    "ReasoningState",
    "LMTrainer",
    "save_checkpoint",
    "load_checkpoint",
]


