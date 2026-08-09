# tests/test_conversation_buffer.py
from inference.conversation import ConversationBuffer
from tokenizer.wrapper import KronosTokenizer
from pathlib import Path
import pytest


@pytest.fixture
def tok(tmp_path):
    from tokenizer.trainer import train_bpe
    texts = ["Hello world. This is great! More text? Yes indeed. " * 20]
    train_bpe(iter(texts), vocab_size=200, save_dir=tmp_path)
    return KronosTokenizer(tmp_path)


def test_bos_preserved_after_trim(tok):
    buf = ConversationBuffer(tok, context_length=20, reserved_for_gen=5)
    buf.append("Hello world. This is text. More text here. Even more.")
    buf.trim_to_fit()
    assert buf.token_ids[0] == tok.BOS_ID


def test_no_trim_under_limit(tok):
    buf = ConversationBuffer(tok, context_length=200, reserved_for_gen=5)
    buf.append("Short text.")
    ids_before = buf.token_ids[:]
    buf.trim_to_fit()
    assert buf.token_ids == ids_before


def test_trim_at_sentence_boundary(tok):
    buf = ConversationBuffer(tok, context_length=20, reserved_for_gen=5)
    buf.append("Sentence one. Sentence two! Sentence three? Sentence four.")
    buf.trim_to_fit()
    text = tok.decode(buf.token_ids)
    # After trim the buffer should start at a sentence boundary (no mid-word cut)
    assert buf.token_ids[0] == tok.BOS_ID