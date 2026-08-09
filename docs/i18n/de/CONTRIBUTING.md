# Mitwirken

**Sprachen:** [English](../../../CONTRIBUTING.md) · [Español](../es/CONTRIBUTING.md) · **Deutsch** · [中文](../zh/CONTRIBUTING.md) · [Русский](../ru/CONTRIBUTING.md)

Danke fürs Reinschauen. Dies ist ein kleines Ein-Personen-Projekt, es gibt also keinen
schweren Prozess – aber ein paar Hinweise machen die Sache runder.

## Einrichtung

```bash
git clone <repo-url> kronos-synapse-dialogue
cd kronos-synapse-dialogue
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Das gibt dir den Befehl `kronos_synapse` sowie `pytest` und `mypy`. Es gibt ein `Makefile`
mit Abkürzungen (`make dev`, `make test`, `make typecheck`), falls du die magst.

## Bevor du einen PR öffnest

- Führe die Tests aus: `pytest` (oder `make test`). Sie sind schnell und rein CPU-basiert.
- Führe den Typechecker aus: `mypy .`. Der Code ist vollständig typisiert; bitte halte es so.
- Halte den Stil konsistent mit dem, was schon da ist: 4-Leerzeichen-Einrückung,
  Typannotationen an öffentlichen Funktionen, kurze Kommentare nur dort, wo das *Warum*
  nicht offensichtlich ist. Füge keine Docstrings hinzu, die nur die Signatur wiederholen.
- Wenn du das Modell oder die Packing-Logik berührst, füge einen Test in `tests/` hinzu oder
  aktualisiere ihn.

## Was willkommen ist

- Fehlerbehebungen und klarere Fehlermeldungen.
- Leistungsarbeit am CPU-Trainingspfad.
- Kleine, gut abgegrenzte Funktionen (ein neuer Sampler, eine neue Korpusquelle usw.). Der
  `DialogueDatasetSource`-Stub in `data/sources.py` ist ein naheliegender Erweiterungspunkt.

Für etwas Großes öffne zuerst ein Issue, damit wir den Ansatz besprechen können, bevor du
Zeit investierst.

## Commit-Nachrichten

Kurze Betreffzeile im Imperativ ("Add nucleus sampling clamp", nicht "Added..."), mit einem
Rumpf, falls die Änderung Kontext braucht. Eine logische Änderung pro Commit, wo praktikabel.
