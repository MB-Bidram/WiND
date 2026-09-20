"""Small modality adapters that convert data to/from Wind token tensors."""

from __future__ import annotations

import torch
from torch import nn

from .modules import WindModule


class LinearAdapter(WindModule):
    """Project continuous features with shape ``[..., input_dim]`` to tokens."""

    def __init__(self, input_dim: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TextAdapter(WindModule):
    """Embedding adapter for integer token IDs."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class ImageAdapter(WindModule):
    """Patchify images into ``[batch, tokens, dim]`` tensors."""

    def __init__(self, channels: int, dim: int, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.proj(images)
        return x.flatten(2).transpose(1, 2)


class VideoAdapter(WindModule):
    """Tubelet-patch video adapter producing spatiotemporal token sequences."""

    def __init__(self, channels: int, dim: int, patch_size: int = 16, frames: int = 2):
        super().__init__()
        self.proj = nn.Conv3d(
            channels, dim, kernel_size=(frames, patch_size, patch_size),
            stride=(frames, patch_size, patch_size),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = self.proj(video)
        return x.flatten(2).transpose(1, 2)


class AudioAdapter(WindModule):
    """Patch waveform [batch, channels, samples] to continuous feature tokens.

    Input adapter for conditioning, not a learned music codec or waveform decoder.
    Sample rate/resampling and waveform normalization are the caller's job.
    """
    def __init__(self, channels: int, dim: int, patch_size: int = 320):
        super().__init__()
        if min(channels, dim, patch_size) < 1:
            raise ValueError("channels, dim and patch_size must be positive")
        self.proj = nn.Conv1d(channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.size(-1) < self.proj.kernel_size[0]:
            raise ValueError("expected [batch, channels, samples] with at least one patch")
        return self.proj(waveform).transpose(1, 2)
