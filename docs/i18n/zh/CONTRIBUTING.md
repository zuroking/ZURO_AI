# 参与贡献

**语言：** [English](../../../CONTRIBUTING.md) · [Español](../es/CONTRIBUTING.md) · [Deutsch](../de/CONTRIBUTING.md) · **中文** · [Русский](../ru/CONTRIBUTING.md)

感谢你来看看。这是一个由单人维护的小项目，所以没有繁重的流程——但有几点说明能让事情更顺畅。

## 环境搭建

```bash
git clone <repo-url> kronos-synapse-dialogue
cd kronos-synapse-dialogue
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

这会给你 `kronos_synapse` 命令，外加 `pytest` 和 `mypy`。如果你喜欢，`Makefile` 里也有一些
快捷方式（`make dev`、`make test`、`make typecheck`）。

## 在提交 PR 之前

- 运行测试：`pytest`（或 `make test`）。它们很快，且只用 CPU。
- 运行类型检查器：`mypy .`。代码是完全带类型的；请保持这样。
- 让风格与现有代码保持一致：4 空格缩进、公共函数带类型注解、只在*为什么*不明显之处写简短
  注释。不要添加只是复述签名的 docstring。
- 如果你改动了模型或打包逻辑，请在 `tests/` 中添加或更新一个测试。

## 欢迎哪些贡献

- 缺陷修复和更清晰的错误信息。
- CPU 训练路径上的性能工作。
- 小而界定清晰的功能（一个新的采样器、一个新的语料源等）。`data/sources.py` 中的
  `DialogueDatasetSource` 存根是一个明显的扩展点。

对于任何较大的改动，请先开一个 issue，以便在你投入时间之前先讨论清楚方案。

## 提交信息

简短的祈使句主题行（"Add nucleus sampling clamp"，而不是 "Added..."），如果改动需要背景说明
则附上正文。在可行时，每个提交只做一处逻辑改动。
