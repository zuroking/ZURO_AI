from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: LambdaLR,
    step: int,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
    }, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None,
    scheduler: LambdaLR | None,
) -> dict:
    # optimizer/scheduler are optional so eval can load weights only
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler:
        scheduler.load_state_dict(state["scheduler"])
    return {"step": state["step"], "best_val_loss": state["best_val_loss"]}
