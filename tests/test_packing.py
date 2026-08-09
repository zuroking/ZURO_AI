from __future__ import annotations

from pathlib import Path

import numpy as np

from config import DataConfig
from data.packing import pack_corpus
from data.sources import GutenbergCorpusSource
from tokenizer.trainer import train_bpe
from tokenizer.wrapper import KronosTokenizer


def test_pack_preserves_tokens(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text("Hello world. This is great. " * 50)
    (raw / "b.txt").write_text("Another book here. More text. " * 50)
    tok_dir = tmp_path / "tok"
    cfg = DataConfig(
        raw_dir=raw,
        processed_dir=tmp_path,
        tokenizer_dir=tok_dir,
        split_seed=0,
        val_fraction=0.5,
    )
    train_bpe(
        GutenbergCorpusSource(cfg).iter_documents(), vocab_size=200, save_dir=tok_dir
    )
    tok = KronosTokenizer(tok_dir)
    pack_corpus(GutenbergCorpusSource(cfg), tok, cfg)
    train_arr = np.memmap(tmp_path / "train.bin", dtype=np.uint16, mode="r")
    val_arr = np.memmap(tmp_path / "val.bin", dtype=np.uint16, mode="r")
    assert len(train_arr) > 0 and len(val_arr) > 0
    assert all(t < tok.vocab_size for t in train_arr[:100])
