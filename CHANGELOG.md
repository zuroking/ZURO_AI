# Changelog

**Languages:** **English** · [Español](docs/i18n/es/CHANGELOG.md) · [Deutsch](docs/i18n/de/CHANGELOG.md) · [中文](docs/i18n/zh/CHANGELOG.md) · [Русский](docs/i18n/ru/CHANGELOG.md)

All notable changes to this project are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/), and versions aim for
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- `CONTRIBUTING.md`, `CHANGELOG.md`, `.editorconfig`, and a `Makefile` with common
  developer targets.
- Pinned dependency lists (`requirements.txt`, `requirements-dev.txt`).
- Multilingual documentation under `docs/i18n/` (Spanish, German, Chinese,
  Russian) for the README, contributing guide, and changelog, with a language
  switcher at the top of each file.

### Changed
- Trimmed source comments and docstrings throughout for readability.

## [0.1.0]

Initial release.

### Added
- Hand-written decoder-only GPT (~15.3M params) on `torch.nn` primitives:
  `CausalSelfAttention`, `MLP`, `TransformerBlock`, `MiniGPT`.
- Byte-level BPE tokenizer (`train_bpe`) and a thin `KronosTokenizer` wrapper with
  fixed special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`).
- Data pipeline: Gutenberg corpus source with boilerplate stripping/normalization,
  book-level train/val split, and `uint16` memmap packing.
- CPU-first training loop with gradient accumulation, `bfloat16` autocast, cosine
  schedule with warmup, gradient clipping, checkpointing, and JSONL metric logging.
- KV-cache generation with temperature / top-k / top-p / repetition-penalty sampling.
- `ConversationBuffer` with sentence-boundary trimming for chat.
- `kronos_synapse` CLI: `tokenize`, `train`, `eval`, `chat`.
- Unit tests for model shapes, causal masking, attention/KV-cache, packing, and the
  conversation buffer.
