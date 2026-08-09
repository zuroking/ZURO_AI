from __future__ import annotations

import re

from tokenizer.wrapper import KronosTokenizer

_SENT_END_RE = re.compile(r'(?<=[.!?])\s+')


class ConversationBuffer:
    """Plain-text rolling history. Trims from the front at sentence boundaries."""

    def __init__(self, tok: KronosTokenizer, context_length: int, reserved_for_gen: int = 64) -> None:
        self._tok = tok
        self._limit = context_length - reserved_for_gen
        self._text = ""

    def append(self, text: str) -> None:
        self._text = (self._text + " " + text).strip()

    @property
    def token_ids(self) -> list[int]:
        return self._tok.encode(self._text, add_bos=True)

    def trim_to_fit(self) -> None:
        while len(self.token_ids) > self._limit:
            m = _SENT_END_RE.search(self._text)
            if m is None:
                break  # nothing to cut on without splitting a sentence
            self._text = self._text[m.end():]

    def get_input_ids(self) -> list[int]:
        self.trim_to_fit()
        return self.token_ids
