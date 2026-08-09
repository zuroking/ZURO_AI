"""Dataset that serves sliding windows from a packed .bin token file."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import DataConfig, ModelConfig


class BinaryTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Reads a uint16 memmap and yields (inputs, targets) shifted by one."""

    def __init__(self, path: Path, context_length: int) -> None:
        self._data = np.memmap(path, dtype=np.uint16, mode="r")
        self._ctx = context_length

    def __len__(self) -> int:
        return len(self._data) - self._ctx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = torch.from_numpy(
            self._data[idx : idx + self._ctx + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


def make_dataloader(
    split: str,
    cfg_data: DataConfig,
    cfg_model: ModelConfig,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    path = cfg_data.processed_dir / f"{split}.bin"
    ds = BinaryTokenDataset(path, cfg_model.context_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=False)
