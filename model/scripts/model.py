"""Baseline reconstruction network.

MLP: 9 surface+coord features -> temperature at 15 depths.

This is the honest baseline the spatial architectures (CNN / ViT / autoencoder)
must beat. It has no spatial context, yet captures the strong local surface ->
profile relationship and trains in seconds. Keeping the same input/output
contract means a CNN can be dropped in later without touching the API.
"""
from __future__ import annotations

import torch
from torch import nn


class ProfileMLP(nn.Module):
    def __init__(self, n_features: int = 9, n_depths: int = 15, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, n_depths),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
