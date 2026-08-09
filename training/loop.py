from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

from config import ModelConfig, TrainConfig, DataConfig
from model.gpt import MiniGPT
from data.dataset import make_dataloader
from training.scheduler import make_scheduler
from training.checkpoint import save_checkpoint, load_checkpoint

console = Console()


def _build_optimizer(model: MiniGPT, cfg: TrainConfig) -> torch.optim.AdamW:
    # weight decay on matrices only; biases and LayerNorm gains are left alone
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": cfg.weight_decay},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95),
    )


@torch.no_grad()
def _eval_loss(model: MiniGPT, loader: torch.utils.data.DataLoader, steps: int = 20) -> float:
    model.eval()
    losses: list[float] = []
    for i, (x, y) in enumerate(loader):
        if i >= steps:
            break
        logits, _ = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(
    cfg_model: ModelConfig,
    cfg_train: TrainConfig,
    cfg_data: DataConfig,
    resume_path: Path | None = None,
) -> None:
    torch.set_num_threads(cfg_train.num_threads)
    model = MiniGPT(cfg_model)
    console.print(f"Parameters: {model.num_params:,}")
    if cfg_train.compile_model:
        try:
            model = torch.compile(model)  # type: ignore[assignment]
        except Exception:
            console.print("[yellow]torch.compile unavailable, skipping[/yellow]")
    optimizer = _build_optimizer(model, cfg_train)
    scheduler = make_scheduler(optimizer, cfg_train)
    step, best_val = 0, float("inf")
    if resume_path:
        info = load_checkpoint(resume_path, model, optimizer, scheduler)
        step, best_val = info["step"], info["best_val_loss"]
    train_loader = make_dataloader("train", cfg_data, cfg_model, cfg_train.batch_size)
    val_loader = make_dataloader("val", cfg_data, cfg_model, cfg_train.batch_size, shuffle=False)
    log_path = cfg_train.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model.train()
    optimizer.zero_grad()
    accum_loss = 0.0
    with Progress(TextColumn("{task.description}"), BarColumn(), TimeRemainingColumn()) as prog:
        task = prog.add_task("Training", total=cfg_train.max_iters)
        prog.update(task, advance=step)
        data_iter = iter(train_loader)
        while step < cfg_train.max_iters:
            for micro in range(cfg_train.grad_accum_steps):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    x, y = next(data_iter)
                with torch.autocast("cpu", torch.bfloat16, enabled=cfg_train.use_bf16):
                    logits, _ = model(x)
                    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                (loss / cfg_train.grad_accum_steps).backward()
                accum_loss += loss.item() / cfg_train.grad_accum_steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            prog.update(task, advance=1)
            if step % cfg_train.checkpoint_every == 0:
                val_loss = _eval_loss(model, val_loader)
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": step, "train_loss": accum_loss, "val_loss": val_loss}) + "\n")
                save_checkpoint(cfg_train.checkpoint_dir / f"step_{step}.pt", model, optimizer, scheduler, step, best_val)
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(cfg_train.checkpoint_dir / "best.pt", model, optimizer, scheduler, step, best_val)
                accum_loss = 0.0
