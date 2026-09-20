"""Public ergonomic model container."""

from __future__ import annotations

from torch import nn

from winc.modules import Depth


class Sequential(Depth):
    """Wind's stable sequential syntax.

    Accepts either positional modules or an iterable, mirroring
    ``torch.nn.Sequential`` while remaining a Wind ``Depth`` container.
    """

    def __init__(self, *modules: nn.Module):
        if len(modules) == 1 and isinstance(modules[0], (list, tuple)):
            modules = tuple(modules[0])
        super().__init__(*modules)


