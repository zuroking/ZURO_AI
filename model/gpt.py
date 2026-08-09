"""The model itself: embeddings, a stack of blocks, tied lm_head."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import ModelConfig
from model.attention import KVCache
from model.block import TransformerBlock


class MiniGPT(nn.Module):
    """Small decoder-only transformer. lm_head shares weights with tok_emb."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self) -> None:
        # GPT-2 init: N(0, 0.02) everywhere, then damp the residual projections
        # by 1/sqrt(2*n_layers) so deep stacks don't blow up.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        scale = 1.0 / math.sqrt(2 * self.cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.c_proj.weight, std=0.02 * scale)
            nn.init.normal_(block.mlp.fc2.weight, std=0.02 * scale)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        past_kvs: list[KVCache | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[KVCache] | None]:
        B, T = idx.shape
        # positions continue after whatever's already cached
        cache_len = past_kvs[0][0].size(2) if (past_kvs and past_kvs[0] is not None) else 0
        pos = torch.arange(cache_len, cache_len + T, device=idx.device, dtype=torch.long)

        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        present_kvs: list[KVCache] = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs else None
            x, present_kv = block(x, past_kv=past_kv, use_cache=use_cache)
            if use_cache and present_kv is not None:
                present_kvs.append(present_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, (present_kvs if use_cache else None)
