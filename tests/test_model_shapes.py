from __future__ import annotations

import pytest
import torch

from config import ModelConfig
from model.gpt import MiniGPT


@pytest.fixture
def mini_cfg() -> ModelConfig:
    # tiny config so the tests run fast
    return ModelConfig(vocab_size=100, d_model=32, n_layers=2, n_heads=2, d_ff=64, context_length=16)


def test_output_shape(mini_cfg: ModelConfig) -> None:
    model = MiniGPT(mini_cfg)
    idx = torch.zeros(2, 8, dtype=torch.long)
    logits, _ = model(idx)
    assert logits.shape == (2, 8, 100)


def test_param_count_within_5pct() -> None:
    model = MiniGPT(ModelConfig())
    expected = 15_323_136
    assert abs(model.num_params - expected) / expected < 0.05
