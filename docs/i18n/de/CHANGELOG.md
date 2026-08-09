# Änderungsprotokoll

**Sprachen:** [English](../../../CHANGELOG.md) · [Español](../es/CHANGELOG.md) · **Deutsch** · [中文](../zh/CHANGELOG.md) · [Русский](../ru/CHANGELOG.md)

Alle nennenswerten Änderungen an diesem Projekt werden hier dokumentiert. Das Format folgt
grob [Keep a Changelog](https://keepachangelog.com/), und die Versionen orientieren sich an
[Semantic Versioning](https://semver.org/).

## [Unveröffentlicht]

### Hinzugefügt
- `CONTRIBUTING.md`, `CHANGELOG.md`, `.editorconfig` und ein `Makefile` mit gängigen
  Entwicklungszielen.
- Festgelegte Abhängigkeitslisten (`requirements.txt`, `requirements-dev.txt`).
- Mehrsprachige Dokumentation unter `docs/i18n/` (Spanisch, Deutsch, Chinesisch,
  Russisch) für die README, den Beitragsleitfaden und das Änderungsprotokoll, mit
  einem Sprachumschalter oben in jeder Datei.

### Geändert
- Kommentare und Docstrings im Code für bessere Lesbarkeit gekürzt.

## [0.1.0]

Erste Veröffentlichung.

### Hinzugefügt
- Von Hand geschriebenes reines Decoder-GPT (~15,3 Mio. Parameter) auf `torch.nn`-Primitiven:
  `CausalSelfAttention`, `MLP`, `TransformerBlock`, `MiniGPT`.
- Byte-Level-BPE-Tokenizer (`train_bpe`) und ein schlanker `KronosTokenizer`-Wrapper mit
  festen Spezial-Tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`).
- Datenpipeline: Gutenberg-Korpusquelle mit Entfernen/Normalisieren von Standardtext,
  Aufteilung train/val pro Buch und `uint16`-memmap-Packing.
- CPU-first-Trainingsschleife mit Gradientenakkumulation, `bfloat16`-Autocast,
  Kosinus-Plan mit Warmup, Gradient-Clipping, Checkpointing und JSONL-Metrik-Logging.
- KV-Cache-Generierung mit Temperature- / Top-k- / Top-p- / Repetition-Penalty-Sampling.
- `ConversationBuffer` mit Kürzen an Satzgrenzen für den Chat.
- `kronos_synapse`-CLI: `tokenize`, `train`, `eval`, `chat`.
- Unit-Tests für Modell-Shapes, kausale Maske, Attention/KV-Cache, Packing und den
  Konversationspuffer.
