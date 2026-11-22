"""Visual token reduction and the vision→language projector (roadmap §3.1)."""

import torch
from torch import nn


def pixel_unshuffle_tokens(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Merge each factor×factor block of patch tokens into one token.

    [N, g*g, C] -> [N, (g/f)*(g/f), f*f*C], row-major over blocks, and row-major
    inside each block.
    """
    n, length, c = x.shape
    g = int(length**0.5)
    if g * g != length or g % factor:
        raise ValueError(f"cannot unshuffle {length} tokens (grid {g}) by factor {factor}")
    x = x.view(n, g // factor, factor, g // factor, factor, c)
    x = x.permute(0, 1, 3, 2, 4, 5)
    return x.reshape(n, (g // factor) ** 2, factor * factor * c)


class MLPProjector(nn.Module):
    """[LayerNorm] → Linear → GELU → Linear."""

    def __init__(self, in_dim: int, out_dim: int, layernorm: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim) if layernorm else nn.Identity()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))
