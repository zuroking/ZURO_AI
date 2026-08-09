# Kronos Synapse Dialogue Core

![Python](https://img.shields.io/badge/python-3.12+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Params](https://img.shields.io/badge/params-%7E15.3M-informational)

**Sprachen:** [English](../../../README.md) · [Español](../es/README.md) · **Deutsch** · [中文](../zh/README.md) · [Русский](../ru/README.md)

Ein reines Decoder-GPT-Sprachmodell (~15,3 Mio. Parameter), von Grund auf auf reinen
`torch.nn`-Primitiven geschrieben und vollständig auf der CPU trainiert. Es bringt einen
BPE-Tokenizer, eine memmap-gestützte Trainingspipeline und einen streamenden CLI-Chat rund
um einen KV-Cache-Generator mit. Der gesamte Stack – Modell, Daten, Training, Inferenz –
ist von Hand implementiert, ohne das Gerüst von `transformers` oder `nanoGPT`.

> **Was es ist (und was nicht).** Dieses Modell macht **Textfortsetzung**, kein
> Befolgen von Anweisungen. Es ist auf literarischer Prosa (Project Gutenberg) trainiert,
> nicht auf annotierten Dialogpaaren. Du tippst Text; es setzt ihn fort. Es ist kein
> Rollenspiel-Assistent. Siehe [Chat-Paradigma & Grenzen](#chat-paradigma--grenzen).

---

## Inhaltsverzeichnis

- [Schnellstart](#schnellstart)
- [Höhepunkte](#höhepunkte)
- [Architektur](#architektur)
- [Projektstruktur](#projektstruktur)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Konfiguration](#konfiguration)
- [Daten vorbereiten](#daten-vorbereiten)
- [Training](#training)
- [Evaluierung](#evaluierung)
- [Chat & Inferenz](#chat--inferenz)
- [Chat-Paradigma & Grenzen](#chat-paradigma--grenzen)
- [Tests](#tests)
- [Entwicklung](#entwicklung)
- [Leistungshinweise](#leistungshinweise)
- [Roadmap](#roadmap)
- [FAQ](#faq)
- [Mitwirken](#mitwirken)
- [Lizenz](#lizenz)

## Schnellstart

Wenn du es einfach nur laufen sehen willst, hier der komplette Ablauf von Anfang bis Ende:

```bash
# 1. installieren
pip install -e ".[dev]"

# 2. lege ein paar .txt-Bücher in data/raw/gutenberg/, dann:
kronos_synapse tokenize --corpus-dir data/raw/gutenberg

# 3. trainieren (Ctrl+C ist sicher; später von einem Checkpoint fortsetzen)
kronos_synapse train

# 4. mit ihm reden
kronos_synapse chat checkpoints/best.pt
```

Es gibt auch ein `Makefile`, falls du `make tokenize`, `make train`, `make chat`
bevorzugst. Alles läuft auf der CPU – keine GPU, kein Cloud-Konto, keine API-Schlüssel.

## Höhepunkte

- **Selbst gebauter Transformer.** `CausalSelfAttention`, `MLP`, `TransformerBlock` und
  `MiniGPT` sind direkt auf `nn.Linear`, `nn.LayerNorm`, `nn.Embedding` und `nn.Dropout`
  aufgebaut. Die Attention läuft über `F.scaled_dot_product_attention`, das auf der CPU den
  optimierten fusionierten Kernel wählt.
- **CPU zuerst.** Alles läuft ohne CUDA. Die Trainingsschleife balanciert mit
  Gradientenakkumulation, `bfloat16`-Autocast, konfigurierbarer Thread-Anzahl und einem
  optionalen `torch.compile`-Fallback.
- **KV-Cache-Generierung.** Die Inferenz macht einen Prefill-Durchlauf über den Prompt und
  dekodiert danach Token für Token gegen die gecachten Keys/Values – der Unterschied
  zwischen nutzbarem und unbrauchbarem Chat auf einer Laptop-CPU.
- **Steuerbares Sampling.** Temperature, Top-k, Top-p (Nucleus) und Repetition Penalty
  sind alle als CLI-Flags verfügbar.
- **Typisierte Konfiguration.** Drei `pydantic-settings`-Konfigurationen (`ModelConfig`,
  `TrainConfig`, `DataConfig`) ersetzen magische Dictionaries durch validierte, unveränderliche
  Einstellungen.

## Architektur

Ein Standard-Pre-LN-GPT-Decoder. Die folgende Tabelle ist die Standard-`ModelConfig`.

| Komponente | Wert | Anmerkungen |
|---|---|---|
| `vocab_size` | 12.000 | BPE, von Grund auf trainiert |
| `d_model` | 384 | Embedding-/Hidden-Breite |
| `n_layers` | 6 | Transformer-Blöcke |
| `n_heads` | 6 | Head-Dim = 64 |
| `d_ff` | 1.536 | 4× `d_model`, GELU-MLP |
| `context_length` | 256 | gelernte absolute Positions-Embeddings |
| `dropout` | 0.1 | Embeddings, Attention-Ausgabe, MLP-Ausgabe |
| `lm_head` | an `tok_emb` gebunden | spart ~4,6 Mio. Parameter |
| Attention-Kernel | `F.scaled_dot_product_attention` | fusioniert, kausal im Training |
| Gewichts-Init | GPT-2-Schema | `N(0, 0.02)`, Residual-Projektionen skaliert mit `1/√(2·n_layers)` |

**Parameterbudget (~15,3 Mio.):**

| Komponente | Parameter |
|---|---|
| Token-Embedding (mit `lm_head` gebunden) | 4.608.000 |
| Positions-Embedding | 98.304 |
| 6 × Transformer-Block (Attn + MLP + 2× LayerNorm) | 10.626.048 |
| Finale LayerNorm | 768 |
| **Gesamt** | **≈ 15.333.120** |

Die Zahl wird zu Beginn jedes Trainingslaufs protokolliert, damit du bestätigen kannst,
dass sie innerhalb von ±5 % des Ziels liegt.

## Projektstruktur

```text
kronos-synapse-dialogue/
├── pyproject.toml          # Abhängigkeiten + Konsolen-Einstiegspunkt
├── config.py               # ModelConfig / TrainConfig / DataConfig (pydantic-settings)
├── cli.py                  # Typer-Einstiegspunkt: tokenize / train / eval / chat
├── tokenizer/
│   ├── trainer.py          # BPE-Training (HF tokenizers, ByteLevel)
│   └── wrapper.py          # KronosTokenizer: encode / decode / Dialogzüge
├── model/
│   ├── attention.py        # CausalSelfAttention + KVCache
│   ├── mlp.py              # GELU-Feedforward
│   ├── block.py            # Pre-LN-TransformerBlock
│   └── gpt.py              # MiniGPT (Embeddings + Blöcke + gebundener lm_head)
├── data/
│   ├── sources.py          # TextCorpusSource ABC + GutenbergCorpusSource
│   ├── packing.py          # Korpus → Aufteilung pro Buch → uint16-memmap
│   └── dataset.py          # BinaryTokenDataset + DataLoader-Fabrik
├── training/
│   ├── scheduler.py        # Kosinus-Abfall mit linearem Warmup
│   ├── checkpoint.py        # Speichern / Laden von Modell + Optimierer + Scheduler
│   └── loop.py             # Trainingsschleife, Metriken, Rich-Fortschritt
├── inference/
│   ├── generate.py         # KV-Cache-Sampling-Generator
│   └── conversation.py     # ConversationBuffer (Kürzen an Satzgrenzen)
├── tests/                  # Unit-Tests für Shapes, kausale Maske, Packing, Buffer
└── data/
    ├── raw/gutenberg/      # <-- hier legst du die .txt-Korpusdateien ab
    └── processed/          # generiert: tokenizer.json, train.bin, val.bin
```

## Voraussetzungen

- **Python** 3.12 oder neuer
- **PyTorch** 2.3+ (CPU-Build; `bfloat16`-Autocast wird auf Alder Lake emuliert, spart aber
  trotzdem Speicherbandbreite – es gibt einen fp32-Fallback)
- **Hardware**: eine moderne Mehrkern-CPU. Entwicklungsziel war ein Intel i7-1225U
  (2P+8E, AVX2, kein AVX-512). Keine GPU erforderlich.

## Installation

```bash
git clone <repo-url> kronos-synapse-dialogue
cd kronos-synapse-dialogue
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Das registriert den Konsolenbefehl `kronos_synapse` und holt die Entwicklungswerkzeuge
(`pytest`, `mypy`) mit rein.

## Konfiguration

Alle Einstellungen liegen in [`config.py`](../../../config.py) als unveränderliche
`pydantic-settings`-Modelle. Bearbeite dort die Standardwerte, um Modell oder
Trainingsplan neu abzustimmen.

### `ModelConfig`

| Option | Standard | Beschreibung |
|---|---|---|
| `vocab_size` | `12_000` | BPE-Vokabulargröße |
| `d_model` | `384` | Hidden-Breite |
| `n_layers` | `6` | Transformer-Blöcke |
| `n_heads` | `6` | Attention-Heads |
| `d_ff` | `1536` | innere MLP-Breite |
| `context_length` | `256` | maximale Sequenzlänge |
| `dropout` | `0.1` | Dropout-Wahrscheinlichkeit |

### `TrainConfig`

| Option | Standard | Beschreibung |
|---|---|---|
| `num_threads` | `8` | `torch.set_num_threads` |
| `batch_size` | `16` | Mikro-Batch-Größe |
| `grad_accum_steps` | `8` | effektiver Batch = `16 × 8 = 128` |
| `max_iters` | `50_000` | Optimierer-Schritte |
| `lr` | `3e-4` | Spitzenlernrate |
| `lr_min` | `1e-5` | Kosinus-Untergrenze |
| `warmup_iters` | `200` | lineare Warmup-Schritte |
| `weight_decay` | `0.1` | nur auf 2D-Parameter angewendet |
| `max_grad_norm` | `1.0` | Gradient-Clipping |
| `use_bf16` | `True` | `bfloat16`-Autocast auf der CPU |
| `compile_model` | `False` | optionales `torch.compile` mit fp32-Fallback |
| `checkpoint_every` | `1_000` | Takt für Checkpoint + Val-Evaluierung |
| `checkpoint_dir` | `checkpoints/` | Ausgabeverzeichnis |
| `log_file` | `logs/train.jsonl` | Metrik-Log pro Checkpoint |

### `DataConfig`

| Option | Standard | Beschreibung |
|---|---|---|
| `raw_dir` | `data/raw/gutenberg/` | Eingabe-`.txt`-Korpus |
| `processed_dir` | `data/processed/` | Ausgabe `train.bin` / `val.bin` |
| `tokenizer_dir` | `data/processed/tokenizer/` | Ausgabe `tokenizer.json` |
| `min_quote_line_ratio` | `0.0` | optionaler Dialogdichte-Filter |
| `split_seed` | `42` | Seed für die Aufteilung pro Buch |
| `val_fraction` | `0.10` | zurückgehaltene Bücher (im Training ungesehen) |

## Daten vorbereiten

1. Lege Project-Gutenberg-`.txt`-Dateien in `data/raw/gutenberg/`. (Romane des 19. bis
   frühen 20. Jahrhunderts – Austen, Dickens, Twain usw. – funktionieren gut: viel direkte
   Rede und kurze Wortwechsel.)
2. Trainiere den Tokenizer und packe den Korpus in speicherabgebildete Binär-Splits:

```bash
kronos_synapse tokenize --corpus-dir data/raw/gutenberg
```

Das erzeugt drei Artefakte in `data/processed/`:

- `tokenizer/tokenizer.json` — der BPE-Tokenizer
- `train.bin` — Trainings-Tokens als `uint16`-memmap
- `val.bin` — Validierungs-Tokens (ganze Bücher zurückgehalten, kein Fenster-Leck)

Die Korpusbereinigung passiert automatisch: der Kopf-/Fußzeilen-Standardtext von Project
Gutenberg wird entfernt, der Text wird Unicode-NFKC-normalisiert und Whitespace-Folgen
werden zusammengefasst. Setze `min_quote_line_ratio` über 0, um nur Dokumente mit dichtem
zitiertem Dialog zu behalten.

Spezial-Tokens sind fest: `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`. Es werden keine
Rollen-Tokens (`<user>`/`<bot>`) hinzugefügt – siehe die Anmerkung zum
[Chat-Paradigma](#chat-paradigma--grenzen).

## Training

```bash
kronos_synapse train
```

Die Schleife gibt beim Start die Parameterzahl aus und läuft dann mit einem
Rich-Fortschrittsbalken. Alle `checkpoint_every` Schritte:

- evaluiert sie den Validierungsverlust,
- hängt `{step, train_loss, val_loss}` an `logs/train.jsonl` an,
- schreibt `checkpoints/step_<n>.pt`,
- aktualisiert `checkpoints/best.pt`, wenn sich der Validierungsverlust verbessert hat.

**Fortsetzen** von einem beliebigen Checkpoint (Optimierer- und Scheduler-Zustand werden
wiederhergestellt):

```bash
kronos_synapse train --resume checkpoints/best.pt
```

## Evaluierung

Evaluiere einen Checkpoint auf den zurückgehaltenen Validierungsbüchern:

```bash
kronos_synapse eval checkpoints/best.pt --split val
```

Meldet Cross-Entropy-Verlust und Perplexität (`2^loss`) über den Split.

## Chat & Inferenz

```bash
kronos_synapse chat checkpoints/best.pt --temperature 0.8 --top-p 0.9
```

Die Generierung wird Token für Token ins Terminal gestreamt. Der erste Aufruf füllt den
Prompt vor und baut den KV-Cache; jedes weitere Token wird gegen diesen Cache dekodiert, so
dass lange Unterhaltungen reaktionsschnell bleiben.

| Flag | Standard | Beschreibung |
|---|---|---|
| `--temperature` | `0.8` | schärft (`<1`) oder flacht (`>1`) die Verteilung ab |
| `--top-k` | `50` | behält nur die `k` wahrscheinlichsten Tokens (`0` deaktiviert) |
| `--top-p` | `0.9` | Nucleus-Sampling (`1.0` deaktiviert) |
| `--repetition-penalty` | `1.1` | gewichtet bereits erzeugte Tokens herunter |
| `--max-new-tokens` | `128` | Generierungslimit pro Zug |

Der Konversationspuffer hält einen wachsenden Klartext-Verlauf und kürzt, wenn er sich
`context_length` nähert, von vorne an der nächsten Satzgrenze (nie mitten im Wort) und
bewahrt dabei das führende `<bos>`. Dieses gleitende Fenster *ist* das Kurzzeitgedächtnis –
das Modell liest bei jedem Zug den gesamten Puffer.

## Chat-Paradigma & Grenzen

Dieses Projekt ist bewusst begrenzt und ehrlich darüber, was ein Modell mit 15 Mio.
Parametern aus fortlaufender Prosa lernen kann:

- **Es setzt Text fort, es befolgt keine Anweisungen.** In den Trainingsdaten gibt es keine
  `<user>`/`<bot>`-Züge, also hat das Modell kein Signal für explizites Abwechseln. Was du
  tippst, wird als Anfang einer Passage behandelt; das Modell schreibt, was danach kommt.
- **Kohärenz ist lokal.** Erwarte ~5–6 Sätze mit konsistentem Stil und Entitätsverfolgung
  innerhalb des Kontextfensters. Kohärenz über größere Distanzen ist nicht garantiert.
- **Es ist nur englischsprachig** und literarisch gefärbt, weil der Korpus englische
  Belletristik ist.

Ein sinnvoller Test, dass das Kontextfenster tatsächlich genutzt wird: erwähne einen
Charakternamen früh im Puffer und prüfe, ob sich das Modell 4–6 Sätze später noch darauf
bezieht. Ein `DialogueDatasetSource`-Stub liegt in `data/sources.py` als Erweiterungspunkt
bereit, falls du später auf annotiertem Dialog feinabstimmst.

## Tests

```bash
pytest
```

Die Unit-Tests decken die Teile ab, die am ehesten still versagen:

- `tests/test_model_shapes.py` — Tensor-Shapes des Forward-Passes, Parameterzahl
- `tests/test_attention.py` — Attention-Ausgabe-Shapes und KV-Cache-Verhalten
- `tests/test_causal_mask.py` — kein Leck zu zukünftigen Positionen
- `tests/test_packing.py` — Logik der Aufteilung pro Buch und des Binär-Packings
- `tests/test_conversation_buffer.py` — Kürzen an Satzgrenzen bewahrt `<bos>`

Die Tests nutzen eine winzige Wegwerf-Konfiguration und einen kleinen, im Flug trainierten
Tokenizer, sodass die gesamte Suite in wenigen Sekunden fertig ist und keine
heruntergeladenen Daten braucht.

## Entwicklung

Häufige Aufgaben sind im `Makefile` gekapselt:

```bash
make dev         # editierbare Installation mit Dev-Extras
make test        # pytest -q
make typecheck   # mypy .
make clean       # Caches und Build-Artefakte entfernen
```

Der Code ist vollständig typannotiert und mit `mypy` geprüft. Wenn du Code hinzufügst,
halte ihn typisiert und füge einen Test hinzu, falls du das Modell, den Tokenizer oder die
Packing-Pfade berührst. Der Stil ist einfaches Python mit 4-Leerzeichen-Einrückung;
Kommentare erklären das *Warum*, nicht das *Was*. Siehe [CONTRIBUTING.md](CONTRIBUTING.md)
für die vollständige Übersicht.

## Leistungshinweise

Auf der Entwicklungs-CPU (i7-1225U) sind während des Trainings grob **400–800 Tokens/s** zu
erwarten, was etwa **7–14 Stunden pro Epoche** über einen Korpus von ~20 Mio. Tokens
entspricht. Der Durchsatz skaliert mit `num_threads`, `use_bf16` und
`batch_size × grad_accum_steps`; stimme diese auf deine eigenen Kerne und deinen Speicher
ab. Die Generierung mit KV-Cache ist auf derselben CPU messbar schneller als die Baseline
ohne Cache – dieser Abstand ist der Grund, warum Caching hier zwingend ist.

## Roadmap

Grobe Reihenfolge, keine Versprechen:

- **RoPE-Positionskodierung**, um die gelernten absoluten Positionen zu ersetzen und dem
  Modell zu erlauben, über `context_length` hinaus zu extrapolieren.
- **Optionales Instruction-Fine-Tuning** auf dem Basismodell, das den
  `DialogueDatasetSource`-Stub und echte `<user>`/`<bot>`-Rollen-Tokens verdrahtet.
- **Längerer Kontext** (512+), sobald der Trainingsdurchsatz es zulässt.
- **Ein `sample`-CLI-Befehl** für eine einmalige Fortsetzung ohne die interaktive Schleife.
- **KV-Cache-Verdrängung**, damit Chat-Sitzungen unbegrenzt laufen können, statt Text zu
  kürzen.

## FAQ

**Warum nur CPU?** Weil es darum ging, zu sehen, wie weit ein von Grund auf gebautes Modell
auf einem Laptop ohne GPU kommt. Alles ist um diese Einschränkung herum abgestimmt.

**Warum nicht einfach `transformers` verwenden?** Die Teile von Hand zu bauen ist die ganze
Übung. Du siehst genau, was Attention, der KV-Cache und die Trainingsschleife tun.

**Es antwortet mit Kauderwelsch / wiederholt sich.** Kleine Modelle, die auf wenig Daten
trainiert wurden, tun das. Gib ihm mehr Korpus und mehr Schritte, senke `--temperature` und
erhöhe `--repetition-penalty` etwas. Denk auch daran, dass es Text *fortsetzt* – es
beantwortet keine Fragen (siehe [Chat-Paradigma & Grenzen](#chat-paradigma--grenzen)).

**Kann ich meinen eigenen Text statt Gutenberg verwenden?** Ja. Jede einfache `.txt`-Datei
funktioniert; der Gutenberg-spezifische Schritt ist nur das Entfernen von Kopf-/Fußzeilen,
was bei anderem Text ein No-op ist. Für ein anderes Format füge eine
`TextCorpusSource`-Unterklasse in `data/sources.py` hinzu.

**Wie mache ich es größer/kleiner?** Bearbeite die Standardwerte in `config.py` (`d_model`,
`n_layers`, `n_heads`, `context_length`). Die Parameterzahl wird zu Beginn des Trainings
ausgegeben, damit du prüfen kannst, wo du gelandet bist.

## Mitwirken

Beiträge sind willkommen – siehe [CONTRIBUTING.md](CONTRIBUTING.md) für Einrichtung und
Erwartungen sowie [CHANGELOG.md](CHANGELOG.md) für die Versionshistorie. Für etwas Großes
öffne zuerst ein Issue, damit wir uns über den Ansatz einig werden.

## Lizenz

[MIT](../../../LICENSE) © 2026 Kronos
