"""
Vision-Language projector.

Following LLaVA-style architectures, visual features from the ViT are projected
into the LLM's embedding space via an MLP or a simple linear layer.
"""

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """Two-layer MLP projector as used in LLaVA-1.5."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class LinearProjector(nn.Module):
    """Simple linear projection."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
