"""A single pre-LN transformer decoder block."""
from __future__ import annotations

import torch
import torch.nn as nn

from config import ModelConfig
from model.attention import CausalSelfAttention, KVCache
from model.mlp import MLP


class TransformerBlock(nn.Module):
    """Pre-norm block: x + attn(ln1(x)), then x + mlp(ln2(x))."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_out, present_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present_kv
