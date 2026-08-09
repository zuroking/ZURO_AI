# Contributing

**Languages:** **English** · [Español](docs/i18n/es/CONTRIBUTING.md) · [Deutsch](docs/i18n/de/CONTRIBUTING.md) · [中文](docs/i18n/zh/CONTRIBUTING.md) · [Русский](docs/i18n/ru/CONTRIBUTING.md)

Thanks for taking a look. This is a small, single-author project, so there's no
heavy process — but a few notes make things go smoother.

## Getting set up

```bash
git clone <repo-url> kronos-synapse-dialogue
cd kronos-synapse-dialogue
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

That gives you the `kronos_synapse` command plus `pytest` and `mypy`. There's a
`Makefile` with shortcuts (`make dev`, `make test`, `make typecheck`) if you like
those.

## Before you open a PR

- Run the tests: `pytest` (or `make test`). They're fast and CPU-only.
- Run the type checker: `mypy .`. The codebase is fully typed; please keep it that way.
- Keep the style consistent with what's already there — 4-space indent, type hints
  on public functions, short comments only where the *why* isn't obvious. Don't add
  docstrings that just repeat the signature.
- If you touch the model or packing logic, add or update a test under `tests/`.

## What's welcome

- Bug fixes and clearer error messages.
- Performance work on the CPU training path.
- Small, well-scoped features (a new sampler, a new corpus source, etc.). The
  `DialogueDatasetSource` stub in `data/sources.py` is an obvious extension point.

For anything large, open an issue first so we can talk through the approach before
you spend time on it.

## Commit messages

Short imperative subject line ("Add nucleus sampling clamp", not "Added..."), with
a body if the change needs context. One logical change per commit where practical.
