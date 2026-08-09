from __future__ import annotations

import torch

from config import ModelConfig
from model.attention import CausalSelfAttention


def test_attention_forward_no_cache() -> None:
    cfg = ModelConfig()
    attn = CausalSelfAttention(cfg)
    attn.eval()

    x = torch.randn(2, 10, cfg.d_model)
    output, cache = attn(x, past_kv=None, use_cache=False)

    assert output.shape == (2, 10, cfg.d_model)
    assert cache is None


def test_attention_forward_with_cache() -> None:
    cfg = ModelConfig()
    attn = CausalSelfAttention(cfg)
    attn.eval()

    x = torch.randn(2, 10, cfg.d_model)
    output, cache = attn(x, past_kv=None, use_cache=True)

    assert output.shape == (2, 10, cfg.d_model)
    assert cache is not None
    assert len(cache) == 2
    k_cache, v_cache = cache
    assert k_cache.shape == (2, cfg.n_heads, 10, cfg.d_model // cfg.n_heads)
    assert v_cache.shape == (2, cfg.n_heads, 10, cfg.d_model // cfg.n_heads)


def test_attention_decode_with_past_kv() -> None:
    cfg = ModelConfig()
    attn = CausalSelfAttention(cfg)
    attn.eval()

    x_prefill = torch.randn(2, 10, cfg.d_model)
    _, past_kv = attn(x_prefill, past_kv=None, use_cache=True)

    x_new = torch.randn(2, 1, cfg.d_model)
    output, new_kv = attn(x_new, past_kv=past_kv, use_cache=True)

    assert output.shape == (2, 1, cfg.d_model)
    assert new_kv is not None
    k_new, v_new = new_kv
    # cache should now hold the original 10 positions plus the new one
    assert k_new.shape == (2, cfg.n_heads, 11, cfg.d_model // cfg.n_heads)
    assert v_new.shape == (2, cfg.n_heads, 11, cfg.d_model // cfg.n_heads)


def test_attention_head_dimension() -> None:
    cfg = ModelConfig()
    attn = CausalSelfAttention(cfg)

    assert attn.head_dim * attn.n_heads == cfg.d_model
    assert attn.head_dim == cfg.d_model // cfg.n_heads


def test_attention_training_mode() -> None:
    cfg = ModelConfig()
    attn = CausalSelfAttention(cfg)
    attn.train()

    x = torch.randn(2, 10, cfg.d_model)
    output, _ = attn(x, past_kv=None, use_cache=False)

    assert output.shape == (2, 10, cfg.d_model)
