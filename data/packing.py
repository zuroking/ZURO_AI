"""Turn a corpus into two flat uint16 token files (train.bin / val.bin)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from config import DataConfig
from data.sources import TextCorpusSource
from tokenizer.wrapper import KronosTokenizer


def pack_corpus(source: TextCorpusSource, tok: KronosTokenizer, cfg: DataConfig) -> None:
    """Tokenize everything and split by book (not by window) into train/val.

    Splitting whole books keeps validation windows from leaking into training.
    """
    docs = list(source.iter_documents())
    rng = random.Random(cfg.split_seed)
    rng.shuffle(docs)
    n_val = max(1, int(len(docs) * cfg.val_fraction))
    val_docs, train_docs = docs[:n_val], docs[n_val:]

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    _write_split(train_docs, tok, cfg.processed_dir / "train.bin")
    _write_split(val_docs, tok, cfg.processed_dir / "val.bin")


def _write_split(docs: list[str], tok: KronosTokenizer, path: Path) -> None:
    all_ids: list[int] = []
    for doc in docs:
        all_ids.extend(tok.encode(doc, add_bos=True))
        all_ids.append(tok.EOS_ID)
    arr = np.array(all_ids, dtype=np.uint16)
    fp = np.memmap(path, dtype=np.uint16, mode="w+", shape=(len(arr),))
    fp[:] = arr
    fp.flush()
