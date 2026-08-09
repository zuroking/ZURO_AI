# Registro de cambios

**Idiomas:** [English](../../../CHANGELOG.md) · **Español** · [Deutsch](../de/CHANGELOG.md) · [中文](../zh/CHANGELOG.md) · [Русский](../ru/CHANGELOG.md)

Todos los cambios notables de este proyecto se documentan aquí. El formato sigue a grandes
rasgos [Keep a Changelog](https://keepachangelog.com/), y las versiones aspiran al
[versionado semántico](https://semver.org/).

## [Sin publicar]

### Añadido
- `CONTRIBUTING.md`, `CHANGELOG.md`, `.editorconfig` y un `Makefile` con objetivos comunes
  de desarrollo.
- Listas de dependencias fijadas (`requirements.txt`, `requirements-dev.txt`).
- Documentación multilingüe en `docs/i18n/` (español, alemán, chino, ruso) para el
  README, la guía de contribución y el registro de cambios, con un selector de
  idioma en la parte superior de cada archivo.

### Cambiado
- Se recortaron los comentarios y docstrings del código para mejorar la legibilidad.

## [0.1.0]

Versión inicial.

### Añadido
- GPT solo-decodificador escrito a mano (~15,3 M de parámetros) sobre primitivas de
  `torch.nn`: `CausalSelfAttention`, `MLP`, `TransformerBlock`, `MiniGPT`.
- Tokenizador BPE a nivel de byte (`train_bpe`) y un fino envoltorio `KronosTokenizer` con
  tokens especiales fijos (`<pad>`, `<unk>`, `<bos>`, `<eos>`).
- Canalización de datos: fuente de corpus Gutenberg con eliminación/normalización de
  texto repetitivo, división train/val por libro y empaquetado en memmap `uint16`.
- Bucle de entrenamiento CPU-first con acumulación de gradiente, autocast en `bfloat16`,
  programa coseno con warmup, recorte de gradiente, checkpointing y registro de métricas en
  JSONL.
- Generación con caché KV y muestreo por temperatura / top-k / top-p / penalización por
  repetición.
- `ConversationBuffer` con recorte por límite de frase para el chat.
- CLI `kronos_synapse`: `tokenize`, `train`, `eval`, `chat`.
- Pruebas unitarias para shapes del modelo, máscara causal, atención/caché KV, packing y el
  buffer de conversación.
