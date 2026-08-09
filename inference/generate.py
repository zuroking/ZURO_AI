from __future__ import annotations

from typing import Iterator

import torch
import torch.nn.functional as F

from model.gpt import MiniGPT


def generate(
    model: MiniGPT,
    prompt_ids: list[int],
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> Iterator[int]:
    """Autoregressive sampling with a KV cache. Yields one token id at a time."""
    model.eval()
    device = next(model.parameters()).device

    # prefill the whole prompt once, then decode a single token per step
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    _, past_kvs = model(idx, use_cache=True)
    generated = list(prompt_ids)

    for _ in range(max_new_tokens):
        next_idx = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        logits, past_kvs = model(next_idx, past_kvs=past_kvs, use_cache=True)
        logits = logits[0, -1, :] / max(temperature, 1e-6)

        for tok_id in set(generated):
            logits[tok_id] /= repetition_penalty

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_tok)  # type: ignore[arg-type]
        yield next_tok  # type: ignore[misc]
