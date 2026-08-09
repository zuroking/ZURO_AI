"""Thin convenience layer over a saved HF tokenizer."""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer


class KronosTokenizer:
    """Loads tokenizer.json and adds encode/decode helpers with fixed special ids."""

    PAD_ID: int = 0
    UNK_ID: int = 1
    BOS_ID: int = 2
    EOS_ID: int = 3

    def __init__(self, tokenizer_dir: Path) -> None:
        self._tok: Tokenizer = Tokenizer.from_file(
            str(tokenizer_dir / "tokenizer.json")
        )

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids: list[int] = self._tok.encode(text).ids
        return ([self.BOS_ID] if add_bos else []) + ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    def encode_dialogue_turns(self, turns: list[str]) -> list[int]:
        """One sequence: <bos> turn <eos> turn <eos> ..."""
        result: list[int] = [self.BOS_ID]
        for turn in turns:
            result.extend(self._tok.encode(turn).ids)
            result.append(self.EOS_ID)
        return result
