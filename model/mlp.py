"""GELU feed-forward used inside each transformer block."""
from __future__ import annotations

import torch
import torch.nn as nn

from config import ModelConfig


class MLP(nn.Module):
    """d_model -> d_ff -> d_model with a GELU in the middle."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(cfg.dropout)
        self._init_weights(cfg.n_layers)

    def _init_weights(self, n_layers: int) -> None:
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        # fc2 feeds the residual stream, so damp it (see GPT-2 init)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02 / (2 * n_layers) ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))
