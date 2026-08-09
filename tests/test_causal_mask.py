from __future__ import annotations

import torch

from config import ModelConfig
from model.gpt import MiniGPT


def test_causal_mask() -> None:
    # changing token 5 must not affect logits at positions 0-4
    cfg = ModelConfig(vocab_size=100, d_model=32, n_layers=2, n_heads=2, d_ff=64, context_length=16)
    model = MiniGPT(cfg).eval()
    idx = torch.randint(0, 100, (1, 8))
    logits1, _ = model(idx)
    idx2 = idx.clone()
    idx2[0, 5] = (idx2[0, 5] + 1) % 100
    logits2, _ = model(idx2)
    assert torch.allclose(logits1[0, :5], logits2[0, :5], atol=1e-5)
