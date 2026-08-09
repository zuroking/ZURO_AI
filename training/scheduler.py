from __future__ import annotations

import math

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from config import TrainConfig


def make_scheduler(optimizer: optim.Optimizer, cfg: TrainConfig) -> LambdaLR:
    """Linear warmup, then cosine decay down to lr_min/lr of the peak."""

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_iters:
            return step / max(1, cfg.warmup_iters)
        progress = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = cfg.lr_min / cfg.lr
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
