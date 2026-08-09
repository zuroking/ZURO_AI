# Convenience targets. `make` on its own prints the list.
.DEFAULT_GOAL := help
.PHONY: help install dev test typecheck tokenize train eval chat clean

help:
	@echo "install    - pip install -e . (runtime only)"
	@echo "dev        - pip install -e .[dev] (adds pytest, mypy)"
	@echo "test       - run the test suite"
	@echo "typecheck  - run mypy"
	@echo "tokenize   - train tokenizer + pack corpus (CORPUS=path, default data/raw/gutenberg)"
	@echo "train      - start a training run (RESUME=path optional)"
	@echo "eval       - evaluate a checkpoint (CKPT=path, default checkpoints/best.pt)"
	@echo "chat       - open a chat session (CKPT=path, default checkpoints/best.pt)"
	@echo "clean      - remove caches and build artifacts"

CORPUS ?= data/raw/gutenberg
CKPT   ?= checkpoints/best.pt

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

typecheck:
	mypy .

tokenize:
	kronos_synapse tokenize --corpus-dir $(CORPUS)

train:
	kronos_synapse train $(if $(RESUME),--resume $(RESUME),)

eval:
	kronos_synapse eval $(CKPT) --split val

chat:
	kronos_synapse chat $(CKPT)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache *.egg-info build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
