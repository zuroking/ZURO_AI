from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig

KVCache = tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention on top of F.scaled_dot_product_attention.

    Handles both the training/prefill path and single-token decode with a KV cache.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must divide evenly by n_heads"
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self._init_weights(cfg.n_layers)

    def _init_weights(self, n_layers: int) -> None:
        nn.init.normal_(self.c_attn.weight, mean=0.0, std=0.02)
        # c_proj is a residual projection, so scale it down like GPT-2 does
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02 / (2 * n_layers) ** 0.5)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        split = lambda t: t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present: KVCache | None = (k, v) if use_cache else None
        # Only mask when there's no cache (training/prefill). During decode the
        # lone new token must see the whole cached history, so is_causal=False.
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
            is_causal=(past_kv is None),
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y)), present
