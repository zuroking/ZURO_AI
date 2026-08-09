"""Train a byte-level BPE tokenizer with HF tokenizers and dump it to disk."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# order matters: these map to fixed ids 0..3 in KronosTokenizer
SPECIAL_TOKENS: list[str] = ["<pad>", "<unk>", "<bos>", "<eos>"]


def train_bpe(
    text_iter: Iterator[str],
    vocab_size: int,
    save_dir: Path,
) -> Tokenizer:
    """Train on text_iter and write save_dir/tokenizer.json."""
    tokenizer: Tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(text_iter, trainer=trainer)
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_dir / "tokenizer.json"))
    return tokenizer
