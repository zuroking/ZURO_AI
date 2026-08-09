"""Where training text comes from. Right now that's plain Gutenberg .txt files."""
from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Iterator

from config import DataConfig

_HEADER_RE = re.compile(r"\*{3}\s*START OF THE PROJECT GUTENBERG EBOOK .+?\*{3}", re.IGNORECASE | re.DOTALL)
_FOOTER_RE = re.compile(r"\*{3}\s*END OF THE PROJECT GUTENBERG EBOOK .+?\*{3}", re.IGNORECASE | re.DOTALL)


class TextCorpusSource(ABC):
    """Anything that can hand us a stream of cleaned documents."""

    @abstractmethod
    def iter_documents(self) -> Iterator[str]:
        ...


class GutenbergCorpusSource(TextCorpusSource):
    """Reads .txt files from raw_dir, strips PG boilerplate, normalizes text.

    If min_quote_line_ratio is set, drops books without enough dialogue.
    """

    def __init__(self, cfg: DataConfig) -> None:
        self._raw_dir = cfg.raw_dir
        self._min_ratio = cfg.min_quote_line_ratio

    def iter_documents(self) -> Iterator[str]:
        for path in sorted(self._raw_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="replace")
            text = _strip_gutenberg(text)
            text = _normalize(text)
            if self._min_ratio > 0.0 and _quote_ratio(text) < self._min_ratio:
                continue
            yield text


def _strip_gutenberg(text: str) -> str:
    """Cut everything before the START marker and after the END marker."""
    m = _HEADER_RE.search(text)
    text = text[m.end():] if m else text
    m = _FOOTER_RE.search(text)
    return text[:m.start()] if m else text


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _quote_ratio(text: str) -> float:
    """Fraction of lines that contain a quotation mark."""
    lines = text.splitlines()
    if not lines:
        return 0.0
    quoted = sum(1 for ln in lines if '"' in ln or '"' in ln or '"' in ln)
    return quoted / len(lines)


class DialogueDatasetSource(TextCorpusSource):
    """Placeholder for loading annotated dialogue (JSONL) once we fine-tune."""

    def iter_documents(self) -> Iterator[str]:
        raise NotImplementedError("DialogueDatasetSource is reserved for future fine-tuning")
